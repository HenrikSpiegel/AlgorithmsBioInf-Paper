
import numpy as np
import matplotlib.pyplot as plt


class hmm_decoder():
    
    def __init__(self, sequence, outfile=None, verbose=False, state=0):
        #try:
        from hmm_conf import states, transition_matrix, symbols, emission_probs, initial_prob, state_names
        self.states = states
        self.state_names = state_names
        self.transition_matrix = transition_matrix
        self.symbols = symbols
        self.emission_probs = emission_probs
        self.initial_prob = initial_prob
        self.p_state = state
        
        self.logged = False #Flag for whether probs are log transformed
        #except:
         #   print("Config file not probably configured")
         #   sys.exit
        self.filename = outfile
        self.verbose = verbose
        self.set_sequence(sequence)
        
    def set_sequence(self, seq):
        self.sequence = seq
        self.encode_seq()
    
    def encode_seq(self):
        if self.sequence is None:
            raise Exception ("No sequence set")
    
        enc = [0] * len(self.sequence)

        for i in range(len(self.sequence)):
            enc[i] = self.symbols.find(self.sequence[i])

        self.encoded_sequence = enc
        
    def log_probs(self):
        self.logged = True
        self.initial_prob = np.log10(self.initial_prob) 
        self.emission_probs  = np.log10(self.emission_probs)
        self.transition_matrix = np.log10(self.transition_matrix)
    
    def pwr_probs(self):
        self.logged = False
        self.initial_prob = np.power(10, self.initial_prob) 
        self.emission_probs  = np.power(10, self.emission_probs)
        self.transition_matrix = np.power(10, self.transition_matrix)
    
    def viterbi_init(self):
        if not self.logged:
            self.log_probs()
        if not hasattr(self, "encoded_sequence"):
            self.encode_seq()

        delta = np.zeros(shape=(self.states, len(self.encoded_sequence)))

        arrows = np.ndarray(shape=(self.states, len(self.encoded_sequence)), dtype=object)

        # initial conditions
        for i in range(0, self.states):

            delta[i][0] = self.initial_prob[i]+self.emission_probs[i][self.encoded_sequence[0]] # Remember we work in log space 

            arrows[i][0] = 0

        self.delta = delta
        self.arrows = arrows
    
    def viterbi_calc_delta(self):
        self.viterbi_init()
        # main loop
        for i in range(1, len(self.encoded_sequence)):

            for j in range(0, self.states): #Current state j

                max_arrow_prob = -np.inf # A very low negative number
                max_arrow_prob_state = -1

                for k in range(0, self.states): #Prior state k

                    # arrow_prob is the probability of ending in the state j from the state k
                    arrow_prob = self.delta[k][i-1] + self.transition_matrix[k,j] + self.emission_probs[j][self.encoded_sequence[i]] #

                    if arrow_prob > max_arrow_prob: 
                        max_arrow_prob = arrow_prob #
                        max_arrow_prob_state = k #

                # store prob
                self.delta[j][i] = max_arrow_prob

                # store arrow
                self.arrows[j][i] = max_arrow_prob_state
                
        if self.verbose: print("Delta Matrix: "), print(self.delta)
    
    def viterbi_traceback(self):
        if not hasattr(self, "delta"):
            self.viterbi_calc_delta()
        
        if self.filename is None: 
            fh = None
        else:
            fh = open(self.filename , "w")
        
        path = []

        max_state = np.argmax(self.delta[:, -1]) # Find the index of the max value in the last column of delta
        max_value = self.delta[max_state, -1] # Find the max value in the last column of delta

        print("log(Max_path):", max_value, file=fh)

        print("Seq: ", self.sequence, file=fh)

        path.append(str(max_state))

        old_state = max_state

        for i in range(len(self.encoded_sequence)-2, -1, -1):

            current_state = self.arrows[old_state][i+1]

            path.append(str(current_state))

            old_state = current_state 

        print("Path:", "".join(reversed(path)), file=fh)
        
        if fh: fh.close()
    
    def forward_ini(self):
        if self.logged:
            self.log_probs()
        if not hasattr(self, "encoded_sequence"):
            self.encode_seq()

        alpha = np.zeros(shape=(self.states, len(self.encoded_sequence)))

        for i in range(0, self.states): 

            alpha[i][0] = self.initial_prob[i]*self.emission_probs[i][self.encoded_sequence[0]]

        self.alpha = alpha
    
    def backward_ini(self):
        if not hasattr(self, "encoded_sequence"):
            self.encode_seq()
       
        beta = np.zeros(shape=(self.states, len(self.encoded_sequence)))

        for i in range(0, self.states):

            beta[i][-1] = 1

        self.beta = beta
        
    def forward_run(self):
        self.forward_ini()
        alpha = self.alpha
        
        # main loop
        for i in range(1, len(self.encoded_sequence)):

            for j in range(0, self.states):
                #Sum of probabilities for all states in the prior position
                _sum = sum([alpha[k][i-1]* self.transition_matrix[k,j] for k in range(self.states)]) 
                # store prob
                alpha[j][i] = self.emission_probs[j][self.encoded_sequence[i]] * _sum
        self.alpha = alpha


    def backward_run(self):
        self.backward_ini()
            
        for i in range(len(self.encoded_sequence)-2, -1, -1):

            for j in range(0, self.states):

                _sum = 0

                for k in range(0, self.states):

                    _sum += self.emission_probs[k][self.encoded_sequence[i+1]] * self.transition_matrix[j,k] * self.beta[k][i+1]

                # store prob
                self.beta[j][i] = _sum



    def calc_posterior(self):
        self.forward_run()
        self.backward_run()
            
        p_state = self.p_state
        
        
        posterior = np.zeros(shape=(len(self.encoded_sequence)), dtype=float)
        

        p_x = 0
        for j in range(0, self.states):
            p_x += self.alpha[j][-1]
            
        if self.filename is None: 
            fh = None
        else:
            fh = open(self.filename , "w")

        print ("Log(Px):", np.log(p_x), file=fh)

        for i in range(0, len(self.encoded_sequence)):

            posterior[i] = (self.alpha[p_state][i]*self.beta[p_state][i])/p_x # p = (f_i * b_i)/p_x

            print ("Posterior", i, self.sequence[i], self.encoded_sequence[i], np.log(self.alpha[p_state, i]), np.log(self.beta[p_state, i]), posterior[i], file=fh)
        self.posterior = posterior
    
    def plot_posterior(self):
        if not hasattr(self, "posterior"):
            print("posterior missing: Calculating now")
            self.calc_posterior()
        
        plt.bar(x = [x for x in range(len(self.sequence))], 
                height = self.posterior,
                tick_label = [x for x in self.sequence])

        plt.xlabel("Sequence")
        plt.ylabel(f"P(state = {self.state_names[self.p_state]})")

        plt.savefig("posterior.png")
        

        
