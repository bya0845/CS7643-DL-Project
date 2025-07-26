import time
import copy
import pathlib
import os
import importlib
import torch
import torchvision
import torch.nn as nn
import inspect
import torchvision.transforms as transforms
import torch.nn.utils.prune as prune
import mlflow
import tempfile
from torch.utils.data import DataLoader, Dataset, random_split
from accelerate import Accelerator
from datetime import datetime
import torch_pruning as tp


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum/self.count

    
class Solver(object):
    def __init__(self, **kwargs):
        self.accelerator = Accelerator()
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # need to do this on PACE I think if running batch jobs
            torch.cuda.reset_peak_memory_stats()
            self.accelerator.print(f'available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')

        self.path_prefix = kwargs.pop("path_prefix", ".")
        self.batch_size = kwargs.pop("batch_size", 256)
        self.model_type = kwargs.pop("model_name", "None")
        self.scheduler_type = kwargs.pop("scheduler", "cosine")
        self.dataset = kwargs.pop("dataset", "CIFAR10")
        self.optimizer_type = kwargs.pop("optimizer", "sgd")
        self.nesterov = kwargs.pop("nesterov", True)
        self.lr = kwargs.pop("learning_rate", 0.0001)
        self.min_lr = kwargs.pop("min_lr", 0.00001)
        self.momentum = kwargs.pop("momentum", 0.9)
        self.reg = kwargs.pop("reg", 0.0005)
        self.beta = kwargs.pop("beta", 0.9999)
        self.gamma = kwargs.pop("gamma", 1.0)
        self.steps = kwargs.pop("steps", [6, 8])
        self.epochs = kwargs.pop("epochs", 10)
        self.warmup_epochs = kwargs.pop("warmup", 0)

        self.num_classes = self._get_num_classes()
        self.num_layers = kwargs.pop("num_layers", 6)
        self.embedding_dim = kwargs.pop("embedding_dim", 256)
        self.mlp_ratio = kwargs.pop("mlp_ratio", 2)
        self.num_heads = kwargs.pop("num_heads", 4)
        self.num_workers = 8 
        self.img_size = kwargs.pop("img_size", 64)

        self.save_best = kwargs.pop("save_best", True)
        self.output_filename = kwargs.pop("output_filename", self.model_type)
        self.mlflow = kwargs.pop("mlflow", True)
        self.prune = kwargs.pop("prune", False)
        self.prune_unstructured = kwargs.pop("prune_unstructured", 0.0)
        self.prune_structured = kwargs.pop("prune_structured", 0.0)

        self.model = self._load_model()
        self.criterion = nn.CrossEntropyLoss()
        self.param_count = self._get_params()
        self.params_after = 0
        self.size_reduction = 0
        if self.param_count is not None:
            self.accelerator.print(f'TRAINABLE PARAMETERS: {self.param_count:,}') 

        self.best = 0.0
        self.best_cm = None
        self.best_epoch = 0
        self.best_model = None
        self.epoch_times = []
        self.training_start_time = None

        self._load_dataset()
        self._load_optimizer()
        self._load_scheduler()

        self.model, self.optimizer, self.scheduler, self.train_loader, self.val_loader = self.accelerator.prepare(self.model, self.optimizer, self.scheduler, self.train_loader, self.val_loader)

        if self.mlflow:
            self.experiment_name = kwargs.pop("experiment_name", None)
            self.training_info = kwargs.pop("training_info", None)
            self.training_title = kwargs.pop("training_title", None)
            self.run_name = f'CCT_{self.num_heads}heads_{self.embedding_dim}embedding_{self.mlp_ratio}mlp_{self.num_layers}layers' # f'{self.model_type}_{self.batch_size}batches_{self.lr}LR
            self._start_mlflow()

    def _get_params(self):
        if self.accelerator.is_main_process:
            try:
                unwrapped_model = self.accelerator.unwrap_model(self.model)
            except (KeyError, AttributeError):
                unwrapped_model = self.model
            return sum(p.numel() for p in unwrapped_model.parameters() if p.requires_grad)  

    def _start_mlflow(self):
        if self.accelerator.is_main_process:
            try:
                if mlflow.active_run():
                    mlflow.end_run()
                mlflow.set_tracking_uri("./runs")    # saves runs locally
                mlflow.set_experiment(self.experiment_name)
                mlflow.pytorch.autolog(
                    log_models=True,
                    log_every_n_epoch=1,
                    log_every_n_step=None,
                    disable=False,
                    exclusive=False,
                    disable_for_unsupported_versions=False,
                    silent=False)
                
                self.accelerator.print("Starting MLflow run...")
                mlflow.start_run()

                active_run = mlflow.active_run()
                if active_run:
                    run_id = active_run.info.run_id
                    experiment_id = active_run.info.experiment_id
                    self.accelerator.print(
                        f"Run ID: {run_id}\n"
                        f"Experiment ID: {experiment_id}\n"
                    )
                else:
                    self.accelerator.print("Failed to start MLflow run")
                    self.mlflow = False
                    return
                
                mlflow.set_tag("training_info", self.training_info)
                mlflow.set_tag("training_title", self.training_title)
                mlflow.set_tag("mlflow.runName", self.run_name)
                mlflow.log_params({
                    "learning_rate": self.lr,
                    "batch_size": self.batch_size,
                    "optimizer": self.optimizer_type,
                    "scheduler": self.scheduler_type,
                    "reg": self.reg,
                    "epochs": self.epochs,
                    "warmup": self.warmup_epochs,
                    "model": self.model_type,
                    "dataset": self.dataset,
                    "prune_unstructured": self.prune_unstructured,
                    "prune_structured": self.prune_structured,
                    "pruning_enabled": self.prune,
                    "num_layers": self.num_layers,
                    "embedding_dim": self.embedding_dim,
                    "mlp_ratio": self.mlp_ratio,
                    "num_heads": self.num_heads,
                    })
                mlflow.log_metrics({"Initial_Parameter_Count": self.param_count})
                
            except Exception as e:
                self.accelerator.print(f'MLflow setup failed: {e}')
                self.mlflow = False  

    def _load_scheduler(self):
        match self.scheduler_type:
            case 'cosine':
                t_max = max(1, self.epochs - self.warmup_epochs)
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=self.min_lr)
            case 'step':
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.steps, gamma=self.gamma)
            case _:
                raise ValueError(f'Unsupported scheduler, only supports step and cosine (for now)')
        
        if self.warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.1, total_iters=self.warmup_epochs)
            schedulers = [warmup_scheduler, self.scheduler]
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=schedulers, milestones=[self.warmup_epochs]) 

    def _load_optimizer(self):
        optimizer_type = getattr(self, 'optimizer_type', 'sgd').lower()
        match optimizer_type:
            case 'adamw':
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.lr,
                    weight_decay=self.reg,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    amsgrad=False
                )          
            case 'sgd':
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=self.lr,
                    momentum=self.momentum,
                    weight_decay=self.reg,
                    nesterov=getattr(self, 'nesterov', True)
                )  
            case 'rmsprop':
                self.optimizer = torch.optim.RMSprop(
                    self.model.parameters(),
                    lr=self.lr,
                    weight_decay=self.reg,
                    momentum=self.momentum,
                    alpha=0.99,
                    eps=1e-8
                )
            case _:
                raise ValueError(f"Unsupported optimizer, only supports: sgd, adamw, rmsprop")
        
    def _get_num_classes(self):
        match self.dataset:
            case 'CIFAR10':
                return 10
            case 'Food101':
                return 101
            case _:
                raise ValueError(f"Unknown dataset: {self.dataset}")

    def _load_dataset(self):
        match self.dataset:
            case 'CIFAR10':
                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

                train_dataset = torchvision.datasets.CIFAR10(root=os.path.join(self.path_prefix, "data", "cifar10"),
                                                             train=True, download=True, transform=transform_train, )

                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

                self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                               num_workers=self.num_workers, pin_memory=True, persistent_workers=True,
                                               prefetch_factor=2, )
                test_dataset = torchvision.datasets.CIFAR10(root=os.path.join(self.path_prefix, "data", "cifar10"),
                                                            train=False, download=True, transform=transform_test)
                self.val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                                              num_workers=self.num_workers, pin_memory=True,    
                                                              persistent_workers=True, prefetch_factor=2, )
            case 'Food101':
                mean = [0.5503, 0.4447, 0.3403]  # custom values
                std = [0.2547, 0.2581, 0.2624]

                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.0), ratio=(0.75, 1.33),
                                                    interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                    ])

                transform_val = transforms.Compose([
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                    ])

                base_train_dataset = torchvision.datasets.Food101(
                    root=os.path.join(self.path_prefix, 'data'), 
                    split='train', download=True, transform=None
                    )
                
                # 80/20 train/val split
                train_size = int(0.80 * len(base_train_dataset))
                val_size = len(base_train_dataset) - train_size
                # Set seed for parallel training
                torch.manual_seed(42)
                train_dataset, val_dataset = random_split(base_train_dataset, [train_size, val_size])

                class TransformedDataset(Dataset):
                    def __init__(self, subset, transform):
                        self.subset = subset
                        self.transform = transform

                    def __getitem__(self, index):
                        x, y = self.subset[index]
                        if self.transform:
                            x = self.transform(x)
                        return x, y

                    def __len__(self):
                        return len(self.subset)
                    
                train_dataset = TransformedDataset(train_dataset, transform_train)
                val_dataset = TransformedDataset(val_dataset, transform_val)

                test_dataset = torchvision.datasets.Food101(
                    root=os.path.join(self.path_prefix, 'data'), 
                    split='test', 
                    download=True, 
                    transform=transform_val  
                    )

                self.train_loader = DataLoader(
                    dataset=train_dataset, batch_size=self.batch_size, shuffle=True, 
                    num_workers=self.num_workers, pin_memory=True, 
                    persistent_workers=True, prefetch_factor=2)
                
                self.val_loader = DataLoader(
                    dataset=val_dataset, batch_size=self.batch_size, shuffle=False,
                    num_workers=self.num_workers, pin_memory=True, 
                    persistent_workers=True, prefetch_factor=2)
                
                self.test_loader = DataLoader(
                    dataset=test_dataset, batch_size=self.batch_size, shuffle=False,
                    num_workers=self.num_workers, pin_memory=True, 
                    persistent_workers=True, prefetch_factor=2)
            case _:
                raise ValueError(f'Dataset {self.dataset} not supported')

    def _load_model(self): 
        module_path = f"models.{self.model_type}"
        module = importlib.import_module(module_path)
        model_class = getattr(module, self.model_type)
        
        if inspect.isclass(model_class):
            if self.model_type == "cct_6_3x1_32":
                cct_params = {
                                'num_classes': self.num_classes,
                                'num_layers': self.num_layers,           
                                'embedding_dim': self.embedding_dim,   
                                'mlp_ratio': self.mlp_ratio,   
                                'num_heads': self.num_heads,         
                            }
                return model_class(**cct_params)
            else:
                try:
                    return model_class(num_classes=self.num_classes, img_size=self.img_size)
                except TypeError:
                    return model_class(num_classes=self.num_classes)
        else:
            return model_class(num_classes=self.num_classes)

    def _reset(self):
        self.best = 0.0
        self.best_cm = None
        self.best_epoch = 0
        self.best_model = None
        self.epoch_times = []
        self.training_start_time = None

        self._load_optimizer()
        self._load_scheduler()
        self.optimizer, self.scheduler = self.accelerator.prepare(self.optimizer, self.scheduler)
            
    def _prune_model(self):
        timestamp = datetime.now().strftime("%M%S_%f")[:-3]
        temp_filepath = os.path.join(tempfile.gettempdir(), f"{timestamp}.pth")

        if self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.load_state_dict(self.best_model)
            print(f'Initial Parameter Count: {self.param_count:,}')
            trainable_params = self.param_count
            nonzero_params = sum((p != 0).sum().item() for p in unwrapped_model.parameters() if p.requires_grad)
            initial_sparsity = 1 - (nonzero_params / trainable_params)
            print(f'Nonzero trainable parameters: {nonzero_params:,}')
            print(f'Initial Sparsity %: {initial_sparsity:.2%}\n')
            try:
                # only pruning conv2d layer weights
                parameters_to_prune = [(m, 'weight') for m in unwrapped_model.modules()if isinstance(m, nn.Conv2d)]
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=self.prune_unstructured,
                )
                              
                for m in unwrapped_model.modules():
                    if isinstance(m, nn.Conv2d) and hasattr(m, 'weight_mask'):
                        prune.remove(m, 'weight')

                # input dims
                example_inputs = torch.randn(1, 3, self.img_size, self.img_size).to(self.accelerator.device)
                imp = tp.importance.MagnitudeImportance(p=1)
                pruner = tp.pruner.MagnitudePruner(
                    model=unwrapped_model,
                    example_inputs=example_inputs,
                    importance=imp,
                    pruning_ratio=self.prune_structured,
                    ignored_layers=[unwrapped_model.linear],
                )

                pruner.step()

                self.params_after = sum(p.numel() for p in unwrapped_model.parameters() if p.requires_grad)
                self.params_removed = self.param_count - self.params_after
                self.size_reduction = self.params_removed / self.param_count
                print(f'Final Parameter Count: {self.params_after:,}')
                print(f'Size reduction: {self.size_reduction:.2%}') 

                if self.mlflow:
                    mlflow.log_metrics({
                        "Initial_Sparsity_Percent": initial_sparsity * 100,
                        "Percent_Reduction": self.size_reduction * 100,
                        "Final_Parameter_Count": self.params_after
                        })
            except Exception as e:
                self.accelerator.print(f"Pruning failed: {str(e)}")
                self.accelerator.wait_for_everyone()   
                self.accelerator.end_training()
                raise RuntimeError(f"Pruning error: {e}") from e
            
            torch.save(unwrapped_model, temp_filepath)  
        
        self.accelerator.wait_for_everyone()   
        try:
            with self.accelerator.main_process_first():
                pruned_model = torch.load(temp_filepath, map_location=self.accelerator.device, weights_only=False)
            self.model = self.accelerator.prepare(pruned_model)
        except Exception as e:
            raise RuntimeError(f"Could not load pruned model: {e}")
        self.accelerator.wait_for_everyone()   

        if self.accelerator.is_main_process:
            try:
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath) 
                    print(f"Cleaned up temp file {temp_filepath}")
            except Exception as e:
                print(f"Warning: Failed to cleanup temp file: {e}")

    def run(self, checkpoint_path=None, testing=False):
        try:
            # testing on single GPU
            if testing and self.accelerator.is_main_process:
                self._test_model(checkpoint_path)
            # training
            else:
                if self.prune:
                    self._train(pre_prune=True)
                    self.accelerator.wait_for_everyone()  # barrier
                    self._prune_model()
                    self._reset()
                    self._train(pre_prune=False)
                else:
                    self._train(pre_prune=False)
            self._save_and_log()
        finally:
            self.accelerator.wait_for_everyone()
            self.accelerator.end_training()
            if self.accelerator.is_main_process and self.mlflow:
                mlflow.end_run()
    
    def _save_and_log(self):
        if self.accelerator.is_main_process:
            if self.save_best:
                basedir = pathlib.Path(__file__).parent.resolve()
                save_path = f"{basedir}/checkpoints/{self.output_filename}.pth"
                self.accelerator.save(self.best_model, save_path)
                print(f'Best model saved to {save_path}')
                if self.mlflow:
                    mlflow.log_artifact(save_path, "models")

            if self.prune:
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                trainable_params = sum(p.numel() for p in unwrapped_model.parameters() if p.requires_grad)
                nonzero_params = sum((p != 0).sum().item() for p in unwrapped_model.parameters() if p.requires_grad)
                final_sparsity = 1 - (nonzero_params / trainable_params)
                print(f'Total trainable parameters: {trainable_params:,}')
                print(f'Nonzero trainable parameters: {nonzero_params:,}')
                print(f'Size reduction: {self.size_reduction:.2%}') 
                print(f'Final sparsity: {final_sparsity:.2%}\n')
                if self.mlflow:
                    mlflow.log_metrics({"final_sparsity_percent": final_sparsity * 100})

    def _test_model(self, checkpoint_path):
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required for testing mode")
        self.test_loader = self.accelerator.prepare(self.test_loader)
        try:
            state_dict = torch.load(checkpoint_path, map_location=self.accelerator.device)
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.load_state_dict(state_dict)
        except Exception as e:
            self.accelerator.print(f"ERROR: Failed to load checkpoint for testing: {e}")
            return 0.0

        test_accuracy = self._evaluate_single_process(epoch=-1)

        print("\n" + "="*60)
        print(f"Final Test Accuracy on '{os.path.basename(checkpoint_path)}': {test_accuracy:.4f}")
        print("="*60 + "\n")
        
        if self.mlflow:
            mlflow.log_metrics({"final_test_accuracy": test_accuracy * 100})
        return test_accuracy

    def _train(self, pre_prune = True):
        self.training_start_time = time.time()
        for epoch in range(self.epochs):
            epoch_start = time.time()
            train_acc = self._train_step(epoch)
            val_acc, val_cm = self._evaluate(epoch)

            self.scheduler.step()
            if self.accelerator.is_main_process:
                if val_acc > self.best:
                    self.best = val_acc
                    self.best_cm = val_cm
                    self.best_epoch = epoch
                    try:
                        unwrapped_model = self.accelerator.unwrap_model(self.model)
                    except (KeyError, AttributeError):
                        unwrapped_model = self.model

                    self.best_model = copy.deepcopy(unwrapped_model.state_dict())
                
                epoch_time = time.time() - epoch_start
                self.epoch_times.append(epoch_time)
                total_elapsed = time.time() - self.training_start_time
                avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
                estimated_remaining = avg_epoch_time * (self.epochs - epoch - 1)

                if self.mlflow:
                    mlflow.log_metrics({
                        "train_accuracy": train_acc,
                        "val_accuracy": val_acc,
                        "best_accuracy": self.best,
                        "learning_rate": self.optimizer.param_groups[0]['lr'],
                        "epoch_time": epoch_time}, 
                        step=epoch
                    )
                print(f'Epoch {epoch + 1}/{self.epochs} completed in {epoch_time:.2f}s')
                print(f'Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Best Val: {self.best:.4f}')
                print(f'Average epoch time: {avg_epoch_time:.2f}s | Total elapsed: {total_elapsed/60:.1f}min | ETA: {estimated_remaining/60:.1f}min')
                print('-' * 50)
        
        if self.accelerator.is_main_process:
            total_training_time = time.time() - self.training_start_time
            final_avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            
            print("="*60)
            print('TRAINING COMPLETE')
            print("="*60)
            print(f'Total training time: {total_training_time/60:.1f} minutes ({total_training_time/3600:.2f} hours)')
            print(f'Average time per epoch: {final_avg_epoch_time:.2f} seconds')
            print(f'BEST ACCURACY: {self.best:.4f} FROM EPOCH {self.best_epoch}')
            print("=" * 60)

            if self.mlflow and not pre_prune:
                mlflow.log_metrics({
                    f"Best_Validation_Accuracy": self.best,
                    f"Best_Epoch": self.best_epoch,
                    f"Average_Epoch_Time_Sec": final_avg_epoch_time,
                })
            
    def _train_step(self, epoch):
        iter_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        self.model.train()
        for idx, (data, target) in enumerate(self.train_loader):
            start = time.time()
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            self.accelerator.backward(loss)
            self.optimizer.step()
            batch_acc = self._check_accuracy(output, target)
            losses.update(loss.item(), output.shape[0])
            acc.update(batch_acc, output.shape[0])

            iter_time.update(time.time() - start)
            if idx % 10 == 0:
                self.accelerator.print(
                    (
                        "Epoch: [{0}][{1}/{2}]\t"
                        "Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                        "Prec @1 {top1.val:.4f} ({top1.avg:.4f})\t"
                    ).format(
                        epoch,
                        idx,
                        len(self.train_loader),
                        iter_time=iter_time,
                        loss=losses,
                        top1=acc,
                        ))
        
        acc_sum = torch.tensor(float(acc.sum), device=self.accelerator.device)
        acc_count = torch.tensor(float(acc.count), device=self.accelerator.device)
        acc_sum_gathered = self.accelerator.gather(acc_sum).sum()
        acc_count_gathered = self.accelerator.gather(acc_count).sum()
        global_train_acc = (acc_sum_gathered / acc_count_gathered).item()
        return global_train_acc
    
    def _evaluate(self, epoch):
        iter_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        cm = torch.zeros(self.num_classes, self.num_classes, device=self.accelerator.device)
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, target) in enumerate(self.val_loader):
                start = time.time()
                output = self.model(inputs)
                loss = self.criterion(output, target)
                batch_acc = self._check_accuracy(output, target)

                _, preds = torch.max(output, 1)
                for t, p in zip(target.view(-1), preds.view(-1)):
                    cm[t.long(), p.long()] += 1

                losses.update(loss.item(), output.shape[0])
                acc.update(batch_acc, output.shape[0])

                iter_time.update(time.time() - start)
                if batch_idx % 10 == 0:
                    self.accelerator.print(
                        f"Val Epoch: [{epoch}][{batch_idx}/{len(self.val_loader)}]\t"
                        f"Time {iter_time.val:.3f} ({iter_time.avg:.3f})")
  
        cm = self.accelerator.gather(cm)
        acc_sum = torch.tensor(float(acc.sum), device=self.accelerator.device)
        acc_count = torch.tensor(float(acc.count), device=self.accelerator.device)
        acc_sum_gathered = self.accelerator.gather(acc_sum).sum()
        acc_count_gathered = self.accelerator.gather(acc_count).sum()
        global_acc = (acc_sum_gathered / acc_count_gathered).item()

        if self.accelerator.is_main_process:
            cm = cm.sum(dim=0) if cm.dim() > 2 else cm
            cm = cm / cm.sum(1, keepdim=True).clamp(min=1e-6)
            
            # per_cls_acc = cm.diag().detach().cpu().numpy().tolist()
            # for i, acc_i in enumerate(per_cls_acc):
                # print(f"Accuracy of Class {i}: {acc_i:.4f}", flush=True) 
            return global_acc, cm
        else:
            return global_acc, cm
    
    def _evaluate_single_process(self, epoch):
        losses = AverageMeter()
        acc = AverageMeter()

        self.model.eval()

        with torch.no_grad():
            for idx, (data, target) in enumerate(self.test_loader ):
                output = self.model(data)
                loss = self.criterion(output, target)
                batch_acc = self._check_accuracy(output, target)
                
                losses.update(loss.item(), output.shape[0])
                acc.update(batch_acc, output.shape[0])
                
                if idx % 10 == 0:
                    print(f"Test Progress: [{idx}/{len(self.test_loader )}]", end='\r')
        
        print() 
        return acc.avg

    def _check_accuracy(self, output, target):
        batch_size = target.shape[0]
        _, pred = torch.max(output, dim=-1)
        correct = pred.eq(target).sum() * 1.0
        acc = correct / batch_size
        return acc
    

