class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'clover':
            return 'samples/clover'
        elif dataset == 'clover_ext':
            return 'samples/clover_ext'
        elif dataset == 'danish':
            return 'samples/danish'
        elif dataset == 'danish_ext':
            return 'samples/danish_ext'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
        
