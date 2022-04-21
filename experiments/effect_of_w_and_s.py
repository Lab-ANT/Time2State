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

def exp_on_USC_HAD(t2s, win_size, step, verbose=False):
    score_list = []
    for subject in range(1,15):
        for target in range(1,6):
            data, groundtruth = load_USC_HAD(subject, target, data_path)
            data = normalize(data)
            # the true num_state is 13
            t2s.predict(data, win_size, step)
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
    for win_size in [400]:
        ARI_list = []
        params_LSE['in_channels'] = 6
        params_LSE['compared_length'] = win_size
        params_LSE['M'] = 20
        params_LSE['N'] = 5
        params_LSE['nb_steps'] = 40
        train, _ = load_USC_HAD(1, 1, data_path)
        train = normalize(train)
        t2s = Time2State(win_size, 10, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit_encoder(train)
        for step in [10,20,30,40,50,60,70,80,90,100]:
            print('window size: %d, step size: %d' %(win_size, step))
            t2s.set_step(step)
            ari = exp_on_USC_HAD(t2s, win_size, step, verbose=False)
            ARI_list.append(ari)
        print(ARI_list)

if __name__ == '__main__':
    run_exp()