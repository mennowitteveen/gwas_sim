import numpy as np
import scipy as sp
import pdb, sys, pickle
import matplotlib.pylab as plt
import datetime
import scipy.stats

class GWASsim(object):

    def __init__(self, num_samples, num_snps, num_causal, pheno_transform = None, transform_param = 2.0, noise_var = 0.1, genetic_var = 0.1, num_phenotypes = 1, num_differentiated = 0, perc_causal_differentiated = 0.5, MAF = 0.05, Fst = 0.1, diploid = True, add_assoc = True, add_noise = True, add_interactions = False, pop_struct = True):

        self.Z = {}
        self.N = num_samples
        self.Q = num_snps
        self.D = num_phenotypes

        # SNPs
        self.MAF = MAF
        self.Fst = Fst
        self.num_causal = num_causal
        if pop_struct and perc_causal_differentiated*num_causal > num_differentiated:
            print "WARNING! the number of differentiated SNPs is not enough to achieve the desired number of diff. causal SNPs, setting it automatically"
            self.num_differentiated = int(perc_causal_differentiated*num_causal)
            
        self.num_differentiated = num_differentiated
        self.perc_causal_differentiated = perc_causal_differentiated
        if diploid:
            self.chr_copies = 2
        else:
            self.chr_copies = 1

        if not pop_struct and num_differentiated != 0:
            self.num_differentiated = 0 
            
        # Phenotype transformations
        self.pheno_transform = pheno_transform
        self.transform_param = transform_param
        # Variances
        self.noise_var = noise_var
        self.genetic_var = genetic_var

        self.pop_struct = pop_struct
        self.generate_snps()
        self.generate_names()
        
        if add_assoc:
            self.add_associations()
        if add_noise:
            self.add_noise()
        if add_interactions:
            self.add_interactions()

        self.get_phenotype()
        self.generated_at = datetime.datetime.now()

    def get_phenotype(self):
        """
        Generates the phenotype by summing together all the individual variance components.
        Note: the phenotype is not guaranteed to be centered  and scaled, because that 
        functionality does not belong here. It's much better if the individual methods 
        implement it.

        """

        self.Y = np.zeros((self.N, self.D))

        for k in self.Z.keys():
            self.Y += self.Z[k]
            print "Variance due to %s = %.3f" % (k, self.Z[k].var())
            
        if self.pheno_transform != None:
            if self.pheno_transform == "exp_ay":
                self.Y_transformed = np.zeros_like(self.Y)
                self.Y_transformed[:self.N/5.0] = np.exp(self.transform_param * self.Y[:self.N/5.0])
            elif self.pheno_transform == "exp_root":
                self.Y_transformed = np.exp(self.Y)**(1.0/self.transform_param)
            elif self.pheno_transform == "rounding":
                self.Y_transformed = self.Y.copy().round(self.transform_param)
                
    def generate_names(self):
        """
        Generates SNP/sample/phenotype names. The causal SNPs have "causal_" prepended 
        in order to make drawing ROCs easier (no need to dump the ground truth).
        
        """
        
        causal = self.causal.tolist()
        self.sample_names = np.array(["Sample%d" % i for i in range(self.N)])
        self.snps_names = np.array(["causal_snp%d" % i if i in causal else "snp%d" % i for i in range(self.Q)])
        self.pheno_names = np.array(["pheno%d" % i for i in range(self.D)])
        
    def generate_snps(self):
        """
        Generates genotypes with a certain MAF and optionally with population structure. 
        In case of no population structure, they are sampled from a binomial, 
        otherwise from a Beta-Binomial (Balding and Nichols, 1995).
        
        """


        print "Simulating SNPs..."

        # Randomly sample causal SNPs
        self.causal = np.random.permutation(self.Q)[:self.num_causal]
        # Randomly sample causal AND differentiated SNPs
        self.differentiated = self.causal[:int(self.num_differentiated*self.perc_causal_differentiated)]
        # Randomly sample SNPs that are differentiated but not causal
        remaining = self.num_differentiated - len(self.differentiated)
        if remaining > 0:
            diff_filter = (np.ones((self.Q,)) == 1)
            diff_filter[self.causal] = False
            non_causal = np.arange(self.Q)[diff_filter]
            self.differentiated = np.append(self.differentiated, np.random.permutation(non_causal)[:remaining])
        
        self.X = np.zeros((self.N, self.Q))


        # self.X = 1.0*(sp.rand(self.N,self.Q)>self.MAF)
        
        for i in range(self.Q):
            p = np.random.uniform(self.MAF, 1-self.MAF)
            self.X[:,i] = np.random.binomial(self.chr_copies, p, size = self.N)
            # completely superfluous given the p above, but good 
            # for the sanity of mind
            # assert (self.X[:,i] != 0).sum()/float(self.N) > self.MAF

        # if there's no population structure to be added we are done, otherwise add it.
        if self.pop_struct:
            assert self.num_differentiated > 0, "At least one SNP has to be differentiated in order to have population structure!"
            self.generate_snps_popstruct()
            
    def generate_snps_popstruct(self):
        """
        Samples differentiated SNPs from a beta-binomial model with a given F_st
        
        """

        F = self.Fst

        for i in self.differentiated: # for each differentiated snp
            # sample the ancestral allele frequency
            p = np.random.uniform(self.MAF, 0.5)

            # WARNING: assuming only two populations for now
            # sample the subpopulation allele frequency 
            alpha = np.random.beta(p*(1-F)/F,(1-p)*(1-F)/F)
            self.X[0:self.N/2.0, i] = np.random.binomial(self.chr_copies, alpha, size = self.N/2.0)
            alpha = np.random.beta(p*(1-F)/F,(1-p)*(1-F)/F)
            self.X[self.N/2.0:self.N, i] = np.random.binomial(self.chr_copies, alpha, size = self.N/2.0)

                
    def add_covariates(self):
        # print "Simulating covariates..."
        raise NotImplementedError

    def add_noise(self):
        """
        Adds Gaussian noise
        
        """

        print "Adding noise..."
        noise_std = np.sqrt(self.noise_var)
        self.Z['noise'] = noise_std*sp.randn(self.N, self.D)

    def add_associations(self):
        print "Simulating associations..."

        if 0:
            mean = np.array([0.0 for i in range(self.N)])
            K = np.dot(self.X[:, self.causal], self.X[:, self.causal].T)
            sigma_g = self.genetic_var/np.diag(K).var()
            XW = np.random.multivariate_normal(mean, sigma_g*K, self.D).T
        else:
            W = np.zeros((self.Q, self.D))
            W[self.causal] = sp.stats.t.rvs(3, 0.0, 0.1, size=(self.num_causal,1))
            XW = np.dot(self.X, W)
            
        self.Z['associations'] = XW

    def add_epistatic_interactions():
        raise NotImplementedError

    def write(base_filename):
        """
        Writes out the simulation
        """
        pass

    def __str__(self):
        message = "Simulation generated on %s \n\n" % self.generated_at
        message += "Dimensions: \n\t %d samples, %d SNPs (%d causal), %d phenotype(s) \n\n" % (self.N, self.Q, self.num_causal, self.D)
        message += "SNP info: \n\t MAF = %.3f, %d chromosome copies \n\n" % (self.MAF, self.chr_copies)
        if self.pop_struct:
            message += "Population structure: \n\t %d differentiated causal SNPs (%d total differentiated SNPs), F_st = %.3f\n\n" % (int(self.num_causal*self.perc_causal_differentiated), self.num_differentiated, self.Fst)
        message += "Variance components: \n\t noise=%.4f, genetic=%.4f" % (self.noise_var, self.genetic_var)
        if self.pheno_transform != None:
            message += "\n\nPhenotype transformation: \n\t %s, parameter = %.2f" % (self.pheno_transform, self.transform_param)
        return message
            
        

def boxcox(Y):
    Y = Y.copy()
    # Y = Y.flatten()

    Yt = sp.stats.boxcox(Y + np.abs(Y.min()) + 0.01)[0]
    Yt -= Yt.mean()
    Yt /= Yt.std()

    return Yt

    
if __name__ == '__main__':
    sim = GWASsim(2000, 6000, 500, num_differentiated = 5900, genetic_var = 1.0, noise_var = 0.5, pop_struct = True, 
                  Fst = 0.1, diploid = True, add_assoc = True, add_noise = True, MAF = 0.25, pheno_transform = "exp_root",
        transform_param = 0.5)
    print sim
    from panama.core import testing

    X = sim.X.copy()
    Y = sim.Y.copy()
    Y -= Y.mean()
    Y /= Y.std()
    X -= X.mean(axis=0)
    X /= X.std(axis=0)

    K = np.cov(sim.X[:, sim.differentiated])
    covs = np.zeros([sim.N,1])
    pv_ideal = testing.interface(X.copy(), Y.copy(), K, covs, I = None, return_fields=['pv'],
                           parallel = False, jobs = 0, add_mean = True, delta_opt_params = None,
                           Ftest = False)[0]

    K = np.diag([1.0 for i in range(sim.N)])
    pv_linear = testing.interface(X, Y, K, covs, I = None, return_fields=['pv'],
                                  parallel = False, jobs = 0, add_mean = True, delta_opt_params = None,
                                  Ftest = False)[0]


    K = np.cov(sim.X)#[:, sim.differentiated])
    pv_bc_kern = testing.interface(X.copy(), boxcox(sim.Y_transformed), K, covs, I = None, return_fields=['pv'],
                                   parallel = False, jobs = 0, add_mean = True, delta_opt_params = None,
                                   Ftest = False)[0]


    
    
    truth = [1.0 if i.find("causal") != -1 else 0.0 for i in sim.snps_names]
    import sklearn.metrics as metrics
    plt.figure()
    fpr, tpr, thrs = metrics.roc_curve(truth, -pv_ideal.flatten())
    plt.plot(fpr, tpr, label = "Ideal (kernel + untransformed pheno)")
    fpr, tpr, thrs = metrics.roc_curve(truth, -pv_linear.flatten())
    plt.plot(fpr, tpr, label = "Linear regression (no kernel, transformed pheno)")
    fpr, tpr, thrs = metrics.roc_curve(truth, -pv_bc_kern.flatten())
    plt.plot(fpr, tpr, label = "BoxCox (kernel)")
    plt.legend(loc=0)
    # plt.xlim((0.0, 0.2))

    # from panama.utilities import qq
    # from panama.utilities.fdr import estimate_lambda
    # qq.qq_plot(pv_ideal.flatten())
    # qq.qq_plot(pv_linear.flatten())    
    # print "\n\n"
    # print "Kernel", estimate_lambda(pv_ideal.flatten())
    # print "NO Kernel", estimate_lambda(pv_linear.flatten())
