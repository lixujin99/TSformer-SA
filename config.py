class Config(object):
    def __init__(self):
        self.train = 'plane'
        self.save = 'TSformer-SA/parameters/'
        self.path1_people = '...' # The path of Task people
        self.path1_car = '...' # The path of Task car
        self.path1_plane = '...' # The path of Task plane
        self.path2 = '.npz'
        self.name_people = ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14','S15','S16','S17','S18',
                            'S19','S20','S21','S22','S23','S24','S25','S26','S27','S28','S29','S30','S31']
        self.name_car = ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19','S20']
        self.name_plane = ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19','S20']

        #all
        self.N = 1
        self.p = 0.5
        self.d_model = 128
        self.hidden = self.d_model * 4
        self.n_heads= 4  
        self.block = 4
         
        #Temporal view
        self.C = 64
        self.T = 250 
        self.patchsizeh = 64
        self.patchsizew = 5
        
        self.H = self.C // self.patchsizeh
        self.W = self.T // self.patchsizew
        
        #Spectral view
        self.scale = 20
        self.wavename = 'mexh' 
        
        self.batchsize = 256
        self.epoch = 100
        self.patience = 110
        self.lr = 5e-4
        self.smooth = 0.05  
        self.num_class = 2
        self.lr_mae = 5e-4
        self.ratio = 0.5
        self.temperature = 0.2 
        self.channels = 4 
        


