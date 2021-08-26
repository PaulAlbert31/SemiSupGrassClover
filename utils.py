import torch
import torchvision.transforms as transforms
import datasets
import TwoSampler

def make_data_loader(args, no_aug=False, transform=None, **kwargs):   
    if 'irish' in args.dataset:
        mean = (0.41637952, 0.5502375,  0.2436111) 
        std = (0.190736, 0.21874362, 0.15318967)
    elif 'danish' in args.dataset:
        mean = (0.3137, 0.4320, 0.1619)
        std = (0.1614, 0.1905, 0.1325)
    
    size = args.size

    if args.base_da: #Weak data augmentation
        transform_train = transforms.Compose([
            transforms.Resize(size),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]) 
    else: #Stronger data augmentation
        transform_train = transforms.Compose([
            transforms.Resize(size),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    #Labeled images of the Irish dataset
    if args.dataset == "irish":
        trainset, testset = datasets.irish(transform=transform_train, transform_test=transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, **kwargs)
    #Labeled images of the Irish dataset + automatic labels from the semseg
    elif args.dataset == "irish_ext":
        trainset, testset, noisy_indexes, clean_indexes = datasets.irish_ext(transform=transform_train, transform_test=transform_test, autom_only=args.autom_only, no_pertu=args.no_pertu)
        if args.no_TS or args.autom_only:#No two sampler
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        else:
            batch_sampler = TwoSampler.TwoStreamBatchSampler(noisy_indexes, clean_indexes, args.batch_size, int(1*args.batch_size/4))
            train_loader = torch.utils.data.DataLoader(trainset, batch_sampler=batch_sampler, **kwargs)
    #Labeled images from the GrassClover dataset
    elif args.dataset == "danish":
        trainset, testset = datasets.danish(transform=transform_train, transform_test=transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, **kwargs)
    #Labeled images from the GrassClover dataset + automatic labels from the semseg
    elif args.dataset == "danish_ext":
        trainset, testset, noisy_indexes, clean_indexes = datasets.danish_ext(transform=transform_train, transform_test=transform_test)
        batch_sampler = TwoSampler.TwoStreamBatchSampler(noisy_indexes, clean_indexes, args.batch_size, int(1*args.batch_size/4))
        train_loader = torch.utils.data.DataLoader(trainset, batch_sampler=batch_sampler, **kwargs)
    else:
        raise NotImplementedError("Dataset {} in not implemented".format(args.dataset))
    
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)
    
    return train_loader, test_loader
