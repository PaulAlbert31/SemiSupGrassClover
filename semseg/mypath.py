class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'danish':
            return '../samples/danish_semseg/'
        elif dataset == 'irish':
            return '../samples/irish_semseg/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError

