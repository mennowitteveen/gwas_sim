import numpy as np
import scipy as sp
import pdb, sys, pickle
import matplotlib.pylab as plt
import datetime

class GWASsim(object):

    def __init__(self, num_samples, num_snps, num_causal, noise_var = 0.1, genetic_var = 0.1, num_phenotypes = 1, num_differentiated = 0, perc_causal_differentiated = 0.5, MAF = 0.05, Fst = 0.1, diploid = True, add_assoc = True, add_noise = True, add_interactions = False, pop_struct = True):

        self.Z = {}
        self.N = num_samples
        self.Q = num_snps
        self.D = num_phenotypes

        # SNPs
        self.MAF = MAF
        self.Fst = Fst
        self.num_causal = num_causal
        if perc_causal_differentiated*num_causal > num_differentiated:
            print "WARNING! the number of differentiated SNPs is not enough to achieve the desired number of diff. causal SNPs, setting it automatically"
            self.num_differentiated = int(perc_causal_differentiated*num_causal)
            
        self.num_differentiated = num_differentiated
        self.perc_causal_differentiated = perc_causal_differentiated
        
        # Variances
        self.noise_var = noise_var
        self.genetic_var = genetic_var
        
        if diploid:
            self.chr_copies = 2
        else:
            self.chr_copies = 1

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
        
        for i in range(self.Q):
            p = np.random.uniform(self.MAF, 0.5)
            self.X[:,i] = np.random.binomial(self.chr_copies, p, size = self.N)
            # completely superfluous given the p above, but good 
            # for the sanity of mind
            assert (self.X[:,i] != 0).sum()/float(self.N) > self.MAF

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

        W = np.zeros((self.Q, self.D))
        W[self.causal] = np.random.randn(self.num_causal,self.D) * np.sqrt(self.genetic_var)

        self.Z['associations'] = np.dot(self.X, W)

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
        
        return message
            
        

if __name__ == '__main__':
    sim = GWASsim(500, 5000, 100, num_differentiated = 1000, genetic_var = 0.1, pop_struct = True)
    print sim
    from panama.core import testing
    K = np.dot(sim.X, sim.X.T)
    # K = np.diag([1.0 for i in range(sim.N)])
    covs = np.ones([sim.N,1])
    pv = testing.interface(sim.X, sim.Y, K, covs, I = None, return_fields=['pv'],
                           parallel = False, jobs = 0, add_mean = False, delta_opt_params = None,
                           Ftest = True)[0]

    plt.figure()
    plt.hist(pv.flatten(), bins = 30)
