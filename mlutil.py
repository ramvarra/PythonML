'''
Utilities used in ML
'''
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib as mpl
import re
import logging
import tempfile
#import graphviz
import sklearn.tree

# ----------------------------------------------------------------------------------------------------------------------
class MLUtil:

    def set_jupyter_defaults(self):
        logging.basicConfig(format='%(asctime)s %(levelname)s - %(message)s', level=logging.DEBUG)
        from IPython.core.display import display, HTML
        display(HTML("<style>.container { width:100% !important; }</style>"))
        np.set_printoptions(precision=2)
    # ------------------------------------------------------------------------------------------------------------------
    def plot_feature_importances(self, ax, clf, feature_names):
        '''
        Plot bar graph of feature importance
        '''      
        ax.barh(range(len(feature_names)), clf.feature_importances_)
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('Feature Importance')
    
    # ------------------------------------------------------------------------------------------------------------------
    def plot_decision_tree(self, clf, feature_names, class_names):
        '''
        Create graphviz rendering of Decision tree.
        '''
        with tempfile.TemporaryFile(mode='w+') as fd:
            sklearn.tree.export_graphviz(clf, out_file=fd, feature_names=feature_names, 
                                     class_names=class_names, filled=True, impurity=False)
            fd.seek(0)
            dot_graph = fd.read()
        return graphviz.Source(dot_graph)

    # ------------------------------------------------------------------------------------------------------------------
    def get_light_dark_cmaps(self, n=0):
        '''
        Generate ListedColormaps that can be used for plotting ligher and darker objects on plots.
        n: Number of colors in the colormap. If n=0 (default), geenrates a default sized colormaps (about 6) that
        include all matching colornames with prefix of light/dark (e.g. lightcyan, darkcyan)
        :return: cmap_light, cmap_dark (Listed colormaps) of size n.
        '''
        d = {'light': {}, 'dark': {}}
        for name, v in mpl.colors.cnames.items():
            m = re.match(r'(?P<tone>light|dark)(?P<color>.*)$', name)
            if m:
                d[m.group('tone')][m.group('color')] = v

        cmap_dark, cmap_light = [], []
        for c, v in d['dark'].items():
            if c in d['light']:
                cmap_dark.append(v)
                cmap_light.append(d['light'][c])
        if n == 0:
            n = len(cmap_dark)
        cmap_light = cmap_light + ['#FFFFAA', '#AAFFAA', '#AAAAFF', '#EFEFEF']
        cmap_dark = cmap_dark + ['#FFFF00', '#00FF00', '#0000FF', '#000000']
        return map(mpl.colors.ListedColormap, [cmap_light[:n], cmap_dark[:n]])
    # ------------------------------------------------------------------------------------------------------------------           
    def scatter_plot(self, x1, x2, y, class_labels=None, ax=None, xy_labels=None, title=None):
        from matplotlib.colors import BoundaryNorm
        from matplotlib.patches import Patch
        
        cm_light, cm_dark = self.get_light_dark_cmaps()
        if class_labels is None:
            class_labels = np.unique(y)
            
        num_labels = len(class_labels)
        bnorm = BoundaryNorm(np.arange(num_labels+1), ncolors=num_labels)
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10,8))
        ax.scatter(x1, x2, c=y, cmap=cm_dark, norm=bnorm, s=65, alpha=0.40, edgecolor='black', lw=1)
        h = []
        for i in range(num_labels):
            h.append(Patch(color=cm_dark.colors[i], label=str(class_labels[i])))
        ax.legend(handles=h)
        if xy_labels:
            xl, yl = xy_labels
        else:
            xl, yl = ['X1', 'X2']
        ax.set_xlabel(xl, fontsize=16)
        ax.set_ylabel(yl, fontsize=16)
        if title:
            ax.set_title(title, fontsize=20)
        
    # ------------------------------------------------------------------------------------------------------------------           
    def _get_title(self, clf, clf_dict, X_train, X_test, y_train, y_test):
        title = type(clf).__name__
        if clf_dict is not None:
            train_score = clf.score(X_train, y_train)
            test_score = clf.score(X_test, y_test)
            title_clf_params = ",".join(f"{k.capitalize()}={v}"for k, v in clf_dict.items())                  
            title = f'{title}({title_clf_params})\nTrain score = {train_score:.2f}, Test score = {test_score:.2f}'            
        return title
    # ------------------------------------------------------------------------------------------------------------------
    def plot_2class_clf(self, clf, clf_dict, ax, X_train, X_test, y_train, y_test):
        '''
        Plot decision boundary and test data for a 2 class classifier. with provided data.
        :param clf: Classifier (e.g. sklearn.neighbors.KNeighborsClassifier)
        :param clf_dict: params for the clf constructors - e.g. {n_neighbors: 1}
        :param ax: axes where to plot
        :param X_train: Input Training set data - must be of shape (TrainRowsx2)
        :param X_test: Input Test set data - must be of shape (TestRowsx2)
        :param y_train: Target Training set - (TrainRowsx1)
        :param y_test: Target Training set - (TrainRowsx1)
        :return:
        '''
        if X_train is not None:
            assert X_train.shape[1] == 2, "X_train must have 2 features"
            if isinstance(X_train, pd.DataFrame):
                X_train = X_train.as_matrix()
                
        assert X_test.shape[1] == 2, "X_test and X_train must have 2 features"        
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.as_matrix()
        
        if clf_dict is not None:
            clf = clf(**clf_dict)
            clf.fit(X_train, y_train)
            
        x1_min, x1_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
        x2_min, x2_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
        
        points = 150
        xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, points), np.linspace(x2_min, x2_max, points))
        z = clf.predict(np.c_[xx1.ravel(), xx2.ravel()]).reshape(xx1.shape)
        
        title = self._get_title(clf, clf_dict, X_train, X_test, y_train, y_test)
        
        light_cm, dark_cm = self.get_light_dark_cmaps(2)
        ax.pcolormesh(xx1, xx2, z, cmap=light_cm)
        if X_train is not None:
            g1 = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, marker='o', edgecolor='black', cmap=dark_cm, label='Train')
        
        g2 = ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=50, marker='^', edgecolor='black', cmap=dark_cm, label='Test')
        
        ax.set_xlim(xx1.min(), xx1.max())
        ax.set_ylim(xx2.min(), xx2.max())
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_title(title)
        
        patch0 = mpl.patches.Patch(color=dark_cm.colors[0], label='class 0')
        patch1 = mpl.patches.Patch(color=dark_cm.colors[-1], label='class 1')
        
        ax.legend(handles=[patch0, patch1])
        return clf


    def plot_class_regions_for_classifier_subplot(self, clf, clf_dict, ax, X_train, X_test, y_train, y_test, xy_labels, target_names = None, plot_decision_regions = True):

        numClasses = np.amax(y_test) + 1
        color_list_light = ['#FFFFAA', '#EFEFEF', '#AAFFAA', '#AAAAFF']
        color_list_bold = ['#EEEE00', '#000000', '#00CC00', '#0000CC']
        cmap_light = mpl.colors.ListedColormap(color_list_light[0:numClasses])
        cmap_bold  = mpl.colors.ListedColormap(color_list_bold[0:numClasses])
        

        h = 0.03
        k = 0.5
        x_plot_adjust = 0.1
        y_plot_adjust = 0.1
        plot_symbol_size = 50

        x_min = X_test[:, 0].min()
        x_max = X_test[:, 0].max()
        y_min = X_test[:, 1].min()
        y_max = X_test[:, 1].max()
        x2, y2 = np.meshgrid(np.arange(x_min-k, x_max+k, h), np.arange(y_min-k, y_max+k, h))
        
        if clf_dict is not None:
            clf = clf(**clf_dict)
            clf.fit(X_train, y_train)
        
        P = clf.predict(np.c_[x2.ravel(), y2.ravel()])
        P = P.reshape(x2.shape)

        if plot_decision_regions:
            ax.contourf(x2, y2, P, cmap=cmap_light, alpha = 0.8)
        if X_train is not None:
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, s=plot_symbol_size, edgecolor = 'black')
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold, s=plot_symbol_size, marker='^', edgecolor = 'black')
        ax.set_xlim(x_min - x_plot_adjust, x_max + x_plot_adjust)
        ax.set_ylim(y_min - y_plot_adjust, y_max + y_plot_adjust)
        title = self._get_title(clf, clf_dict, X_train, X_test, y_train, y_test)
        ax.set_title(title, fontsize=16)
        if xy_labels:
            ax.set_xlabel(xy_labels[0], fontsize=12)
            ax.set_ylabel(xy_labels[1], fontsize=12)
        

        if target_names is not None:
            legend_handles = []
            for i in range(0, len(target_names)):
                patch = mpl.patches.Patch(color=color_list_bold[i], label=target_names[i])
                legend_handles.append(patch)
            ax.legend(loc=0, handles=legend_handles)        

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    mlu = MLUtil()
    light_cm, dark_cm = mlu.get_light_dark_cmaps()
    assert [len(light_cm.colors), len(dark_cm.colors)] > [2, 2], f"light and dark must have atleast 2 color - got [{len(light_cm)}, {len(dark_cm)}]"
    print(f'Light = {light_cm.colors} Dark = {dark_cm.colors}')
