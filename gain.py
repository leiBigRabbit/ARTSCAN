from utils.kernel import gaussianKernel

class  Gain_field(object):
    def IU_mnkl(self, Amn, Smknl):
        IUmnkl = (Smknl + Amn) 
        return IUmnkl
