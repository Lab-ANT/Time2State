import sys
from TSpy.eval import *
from TSpy.label import *
from TSpy.utils import *
from TSpy.view import *
from TSpy.dataset import *

sys.path.append('./')
from Time2State.time2state import Time2State
from Time2State.adapers import *
from Time2State.clustering import *
from Time2State.default_params import *

data_path = os.path.join(os.path.dirname(__file__), '../data/')

def exp_on_USC_HAD(M, N, verbose=False):
    score_list = []
    params_LSE['in_channels'] = 6
    params_LSE['compared_length'] = 256
    params_LSE['M'] = M
    params_LSE['N'] = N
    params_LSE['nb_steps'] = 40
    train, _ = load_USC_HAD(1, 1, data_path)
    train = normalize(train)
    t2s = Time2State(256, 50, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(train, 256, 50)
    # t2s = Time2State(256, 50, CausalConv_LSE_Adaper(params_LSE), DPGMM(None))
    for subject in range(1,15):
        for target in range(1,6):
            data, groundtruth = load_USC_HAD(subject, target, data_path)
            data = normalize(data)
            # the true num_state is 13
            t2s.predict(data, 256, 50)
            prediction = t2s.state_seq
            ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
            score_list.append(np.array([ari, anmi, nmi]))
            if verbose:
                print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %('s'+str(subject)+'t'+str(target), ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))
    return np.mean(score_list[:,0])

def run_exp():
    # for the case M=1 or N=1, please remove the inter/intra part in the loss function
    # M=1 & N=1 case is essentially untrained.
    for M in [10, 20, 30, 40, 50]:
        ARI_list = []
        for N in [2, 4, 6, 8, 10]:
            print('window size: %d, step size: %d' %(M, N))
            sum_ari = 0
            for i in range(10):
                sum_ari += exp_on_USC_HAD(M, N, verbose=False)
            ARI_list.append(sum_ari/10)
        print(np.round(ARI_list, 4))

if __name__ == '__main__':
    run_exp()
    # exp_on_USC_HAD(10,2, verbose=True)