import os
import torch
import numpy as np
import random
from collections import OrderedDict
import dataloaders
from torch.utils.data import DataLoader
import learners

class Trainer:

    def __init__(self, args, seed, metric_keys, save_keys):

        # process inputs
        self.seed = seed
        self.metric_keys = metric_keys
        self.save_keys = save_keys
        self.log_dir = args.log_dir
        self.batch_size = args.batch_size
        self.workers = args.workers
        self.task_step_size=args.other_split_size

        # for generative models, pre-process data to be 0...1; otherwise, pre-process data to be zero mean, unit variance
        if args.learner_type == 'dgr':
            self.dgr = True
        else:
            self.dgr = False
        
        # model load directory
        if args.load_model_dir is not None:
            self.model_first_dir = args.load_model_dir
        else:
            self.model_first_dir = args.log_dir
        self.model_top_dir = args.log_dir

        # select dataset
        self.top_k = 1
        if args.dataset == 'CIFAR10':
            Dataset = dataloaders.iCIFAR10
            num_classes = 10
            self.dataset_size = [32,32,3]
            args.dataroot_dataset=os.path.join(args.dataroot, 'cifar10')
        elif args.dataset == 'CIFAR100':
            Dataset = dataloaders.iCIFAR100
            num_classes = 100
            self.dataset_size = [32,32,3]
            args.dataroot_dataset=os.path.join(args.dataroot, 'cifar100')
        elif args.dataset == 'ImageNet':
            Dataset = dataloaders.iIMAGENET
            num_classes = 1000
            self.dataset_size = [224,224,3]
            self.top_k = 5
            args.dataroot_dataset=os.path.join(args.dataroot, 'imagenet')
        elif args.dataset == 'ImageNet50':
            Dataset = dataloaders.iIMAGENET
            num_classes = 50
            self.dataset_size = [224,224,3]
            self.top_k = 5
            args.dataroot_dataset=os.path.join(args.dataroot, 'imagenet')
        elif args.dataset == 'TinyImageNet':
            Dataset = dataloaders.iTinyIMNET
            num_classes = 200
            self.dataset_size = [64,64,3]
            args.dataroot_dataset=os.path.join(args.dataroot, 'tiny-imagenet')
        elif args.dataset == 'TinyImageNet100':
            Dataset = dataloaders.iTinyIMNET
            num_classes = 100
            self.dataset_size = [64,64,3]
            args.dataroot_dataset=os.path.join(args.dataroot, 'tiny-imagenet')
        else:
            raise ValueError('Dataset not implemented!')

        # load tasks
        class_order = np.arange(num_classes).tolist()
        class_order_logits = np.arange(num_classes).tolist()
        #shuffle the class order 
        if args.rand_split:
            print('=============================================')
            print('Shuffling....')
            print('pre-shuffle:' + str(class_order))
            if args.dataset == 'ImageNet':
                np.random.seed(1993)
                np.random.shuffle(class_order)
            else:
                random.seed(self.seed)
                random.shuffle(class_order)
            print('post-shuffle:' + str(class_order))
            print('=============================================')
        self.tasks = []
        self.tasks_logits = []
        p = 0
        while p < num_classes and (args.max_task == -1 or len(self.tasks) < args.max_task):
            inc = args.other_split_size if p > 0 else args.first_split_size
            self.tasks.append(class_order[p:p+inc])
            self.tasks_logits.append(class_order_logits[p:p+inc])
            p += inc
        self.num_tasks = len(self.tasks)
        self.task_names = [str(i+1) for i in range(self.num_tasks)]

        # number of tasks to perform
        if args.max_task > 0:
            self.max_task = min(args.max_task, len(self.task_names))
        else:
            self.max_task = len(self.task_names)

        # datasets and dataloaders
        train_transform = dataloaders.utils.get_transform(dataset=args.dataset, phase='train', aug=args.train_aug, dgr=self.dgr)
        gan_transform = dataloaders.utils.get_transform(dataset=args.dataset, phase='train', aug=args.train_aug, dgr=self.dgr, gan=True)
        test_transform  = dataloaders.utils.get_transform(dataset=args.dataset, phase='test', aug=args.train_aug, dgr=self.dgr)
        self.train_dataset = Dataset(args.dataroot_dataset, train=True, tasks=self.tasks,
                            download_flag=True, transform=train_transform, 
                            seed=self.seed, validation=args.validation)
        self.test_dataset  = Dataset(args.dataroot_dataset, train=False, tasks=self.tasks,
                                download_flag=False, transform=test_transform, 
                                seed=self.seed, validation=args.validation)

        # save this for E2E baseline
        self.train_dataset.simple_transform = dataloaders.utils.get_transform(dataset=args.dataset, phase='test', aug=args.train_aug, dgr=self.dgr)

        # for oracle
        self.oracle_flag = args.oracle_flag
        self.add_dim = 0

        # Prepare the self.learner (model)
        self.learner_config = {'num_classes': num_classes,
                        'dataset': args.dataset,
                        'lr': args.lr,
                        'momentum': args.momentum,
                        'weight_decay': args.weight_decay,
                        'schedule': args.schedule,
                        'schedule_type': args.schedule_type,
                        'model_type': args.model_type,
                        'model_name': args.model_name,
                        'gen_model_type': args.gen_model_type,
                        'gen_model_name': args.gen_model_name,
                        'optimizer': args.optimizer,
                        'gpuid': args.gpuid,
                        'memory': args.memory,
                        'temp': args.temp,
                        'out_dim': num_classes,
                        'overwrite': args.overwrite == 1,
                        'beta': args.beta,
                        'mu': args.mu,
                        'DW': args.DW,
                        'batch_size': args.batch_size,
                        'power_iters': args.power_iters,
                        'deep_inv_params': args.deep_inv_params,
                        'tasks': self.tasks_logits,
                        'top_k': self.top_k,
                        'log_dir':self.log_dir,
                        'init_generator':args.init_generator,
                        'cgan':args.cgan,
                        'wandb':args.wandb,
                        'disc_lr':args.disc_lr,
                        'supcon':args.supcon,
                        'supcon_temp':args.supcon_temp,
                        'supcon_weight':args.supcon_weight,
                        'dataroot_dataset':args.dataroot_dataset,
                        'gan_transform':gan_transform,
                        'gan_target':args.gan_target,
                        'gan_training':args.gan_training,
                        'wgan_ce':args.wgan_ce,
                        }
        self.learner_type, self.learner_name = args.learner_type, args.learner_name
        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config,Dataset)
        self.learner.print_model()

    def task_eval(self, t_index, local=False):

        val_name = self.task_names[t_index]
        print('validation split name:', val_name)
        
        # eval
        self.test_dataset.load_dataset(t_index, train=True)
        test_loader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
        if local:
            return self.learner.validation(test_loader, task_in = self.tasks_logits[t_index],task_num=-1)
        else:
            return self.learner.validation(test_loader,task_num=-1)

    def train(self, avg_metrics,repeat_idx):
    
        # temporary results saving
        temp_table = {}
        for mkey in self.metric_keys: temp_table[mkey] = []
        temp_dir = self.log_dir + '/temp/'
        if not os.path.exists(temp_dir): os.makedirs(temp_dir)
        visualize_path= os.path.join(self.log_dir,'visualize_weight','repeat-{}'.format(repeat_idx))
        if not os.path.exists(visualize_path): os.makedirs(visualize_path)
        visualize_cm_path=os.path.join(self.log_dir,'visualize_confusion_matrix','repeat-{}'.format(repeat_idx))
        if not os.path.exists(visualize_cm_path): os.makedirs(visualize_cm_path)
        visualize_ml_path=os.path.join(self.log_dir,'visualize_marginal_likelihood','repeat-{}'.format(repeat_idx))
        if not os.path.exists(visualize_ml_path): os.makedirs(visualize_ml_path)

        # for each task
        for task_num in range(self.max_task):

            # save current task index
            self.current_t_index = task_num

            # set seeds
            random.seed(self.seed*100 + task_num)
            np.random.seed(self.seed*100 + task_num)
            torch.manual_seed(self.seed*100 + task_num)
            torch.cuda.manual_seed(self.seed*100 + task_num)
            torch.random.manual_seed(self.seed*100 + task_num)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # print name
            train_name = self.task_names[task_num]
            print('======================', train_name, '=======================')

            # load dataset for task
            task = self.tasks_logits[task_num]
            if self.oracle_flag:
                self.train_dataset.load_dataset(task_num, train=False)
                self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)
                self.add_dim += len(task)
            else:
                self.train_dataset.load_dataset(task_num, train=True)
                self.add_dim = len(task)

            # tell learner number of tasks we are doing
            self.learner.max_task = self.max_task

            # add valid class to classifier
            self.learner.add_valid_output_dim(self.add_dim)

            # load dataset with memory
            self.train_dataset.append_coreset(only=False)#,learner_name=self.learner_name)

            # load dataloader
            train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=int(self.workers))

            # learn
            self.test_dataset.load_dataset(task_num, train=False)
            test_loader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
            if task_num == 0:
                model_save_dir = self.model_first_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[task_num]+'/'
            else:
                model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[task_num]+'/'
            if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
            avg_train_time = self.learner.learn_batch(train_loader, self.train_dataset, model_save_dir, test_loader, task_num)

            # save model
            self.learner.save_model(model_save_dir)
            self.learner.visualize_weight(visualize_path, self.current_t_index)
            self.learner.visualize_confusion_matrix(test_loader,visualize_cm_path, self.current_t_index)
            self.learner.visualize_marginal_likelihood(test_loader,visualize_ml_path, self.current_t_index)
            
            # evaluate acc
            acc_table = []
            self.reset_cluster_labels = True
            for j in range(task_num+1):
                acc_table.append(self.task_eval(j))
            temp_table['acc'].append(np.mean(np.asarray(acc_table)))

            # save temporary results
            for mkey in self.metric_keys:
                save_file = temp_dir + mkey + '.csv'
                np.savetxt(save_file, np.asarray(temp_table[mkey]), delimiter=",", fmt='%.2f')  
            if avg_train_time is not None: avg_metrics['time']['global'][task_num] = avg_train_time
            avg_metrics['mem']['global'][:] = self.learner.count_memory(self.dataset_size)
            if (self.learner_config['dataset']== 'TinyImageNet100' and (self.current_t_index+1)*self.task_step_size==100):
                break
            elif self.learner_config['dataset']== 'ImageNet50' and ((self.current_t_index+1)*self.task_step_size==100):
                break
            else:
                pass

        return avg_metrics 
    
    def summarize_acc(self, acc_dict, acc_table, acc_table_pt):

        # unpack dictionary
        avg_acc_all = acc_dict['global']
        avg_acc_pt = acc_dict['pt']
        avg_acc_pt_local = acc_dict['pt-local']

        # Calculate average performance across self.tasks
        # Customize this part for a different performance metric
        avg_acc_history = [0] * self.max_task
        for i in range(self.max_task):
            train_name = self.task_names[i]
            cls_acc_sum = 0
            for j in range(i+1):
                val_name = self.task_names[j]
                cls_acc_sum += acc_table[val_name][train_name]
                avg_acc_pt[j,i,self.seed] = acc_table[val_name][train_name]
                avg_acc_pt_local[j,i,self.seed] = acc_table_pt[val_name][train_name]
            avg_acc_history[i] = cls_acc_sum / (i + 1)

        # Gather the final avg accuracy
        avg_acc_all[:,self.seed] = avg_acc_history

        # repack dictionary and return
        return {'global': avg_acc_all,'pt': avg_acc_pt,'pt-local': avg_acc_pt_local}

    def evaluate(self, avg_metrics):
        try:
            self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)
        except:
            pass

        # store results
        metric_table = {}
        metric_table_local = {}
        for mkey in self.metric_keys:
            metric_table[mkey] = {}
            metric_table_local[mkey] = {}

        for i in range(self.max_task):

            # load model
            if i == 0:
                model_save_dir = self.model_first_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
            else:
                model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
            self.learner.task_count = i 
            self.learner.add_valid_output_dim(len(self.tasks_logits[i]))
            self.learner.pre_steps()
            self.learner.load_model(model_save_dir)

            # evaluate acc
            metric_table['acc'][self.task_names[i]] = OrderedDict()
            metric_table_local['acc'][self.task_names[i]] = OrderedDict()
            self.reset_cluster_labels = True
            for j in range(i+1):
                val_name = self.task_names[j]
                metric_table['acc'][val_name][self.task_names[i]] = self.task_eval(j)
            for j in range(i+1):
                val_name = self.task_names[j]
                metric_table_local['acc'][val_name][self.task_names[i]] = self.task_eval(j, local=True)
            

        # summarize metrics
        avg_metrics['acc'] = self.summarize_acc(avg_metrics['acc'], metric_table['acc'],  metric_table_local['acc'])

        return avg_metrics
