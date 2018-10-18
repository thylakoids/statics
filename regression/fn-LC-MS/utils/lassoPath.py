from sklearn.linear_model import  LassoLarsIC,LassoLarsCV
from matplotlib import pyplot as plt
import numpy as np
def lARic(X,Y):
    #%% AIC
    # LassoLarsIC: least angle regression with BIC/AIC criterion
    model_bic = LassoLarsIC(criterion='bic')
    model_bic.fit(X, Y)
    alpha_bic_ = model_bic.alpha_

    model_aic = LassoLarsIC(criterion='aic')
    model_aic.fit(X, Y)
    alpha_aic_ = model_aic.alpha_


    def plot_ic_criterion(model, name, color):
        alpha_ = model.alpha_
        alphas_ = model.alphas_
        criterion_ = model.criterion_
        plt.plot(-np.log10(alphas_), criterion_, '--', color=color,
                 linewidth=3, label='%s criterion' % name)
        plt.axvline(-np.log10(alpha_), color=color, linewidth=3,
                    label='alpha: %s estimate' % name)
        plt.xlabel('-log(alpha)')
        plt.ylabel('criterion')

    plt.figure()
    plot_ic_criterion(model_aic, 'AIC', 'b')
    plot_ic_criterion(model_bic, 'BIC', 'r')
    plt.legend()
    plt.title('Information-criterion for model selection')
    plt.show()

def lassolarscv(X,y):
    # LassoLarsCV: least angle regression

    # Compute paths
    model = LassoLarsCV(cv=10).fit(X, y)

    # Display results
    m_log_alphas = -np.log10(model.cv_alphas_)

    plt.figure()
    plt.plot(m_log_alphas, model.mse_path_, ':')
    plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',
             label='Average across the folds', linewidth=2)
    plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
                label='alpha CV')
    plt.legend()

    plt.xlabel('-log(alpha)')
    plt.ylabel('Mean square error')
    plt.title('Mean square error on each fold: Lars')
    plt.axis('tight')
    # plt.ylim(ymin, ymax)

    plt.show()