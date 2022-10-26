"""
Module: Binary label Metrics
About:  Class for computing binay label performance metrics
"""

from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from numbers import Number
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
)


class BinaryLabelMetrics:
    """
    :param num_thresh: Number of thresholds equally spaced between [0,1] at
        which the confusion matrix is computed. Default is 1001. Setting to an
        odd number creates more even splits between 0 and 1.
    :type num_thresh: int, optional
    """

    def __init__(self, num_thresh=1001):
        self._numthresh = num_thresh
        self._modname = list()  # list holding all the added model names
        self._modname_dct = (
            dict()
        )  # dictionary assigning an index to every added model name for efficiency
        self._modname_sz = list()  # name of added models with number of observations
        self._scores = (
            list()
        )  # list holding dataframes containing true vs pred score for each model
        self._confmat = list()  # list holding confusion matrix for each model
        self._auc = list()  # list holding area under ROC curve for each model
        self._f1 = list()  # list holding F1 score for each model
        self._prrec = list()  # list holding average precision for each model
        self._thresh_prev = (
            list()
        )  # threshold at prevalence (prevalence = num of ones/num of obs) for each model

    def add_model(self, name, scores_df, params={}):
        """
        Add model info to internal data structure and compute all neccessary
        calculations. This step takes the longest

        :param name: Model name
        :type name: str
        :param scores_df: Dataframe with columns named 'label' and 'score'
            - label: true labels with ones (events) and zeros (non-events)
            - score: model output; scores between [0,1]
        :type scores_df: :class:`pandas.DataFrame`
        :param params: Dictionary of sklearn parameters
            - skl_auc_average: micro, macro, weighted, samples (http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
            - skl_ap_average:  micro, macro, weighted, samples (http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
        :type params: dict, optional
        """

        # add model name and scores to data structure
        self._modname_dct[name] = len(self._modname)
        self._modname.append(name)
        self._modname_sz.append(f"{name} ({scores_df.shape[0]:,})")
        self._scores.append(scores_df)

        labels = scores_df['label']
        scores = scores_df['score']
        thresholds = np.linspace(
            max(min(scores) - 1e-4, 0.0),
            min(max(scores) + 1e-4, 1.0),
            self._numthresh,
        )

        # calculate confusion matrix at various thresholds
        notlab = 1 - labels
        pos = labels.sum()
        neg = notlab.sum()
        tp, fp, fn, tn = [np.zeros(thresholds.shape[0] + 1) for _ in range(4)]
        for i in range(thresholds.shape[0]):
            tmp = (scores >= thresholds[i]) * 1
            tp[i] = np.einsum("i,i", labels, tmp)
            fp[i] = np.einsum("i,i", notlab, tmp)
            fn[i] = pos - tp[i]
            tn[i] = neg - fp[i]

        # calculate confusion matrix at prevalence threshold
        inds = np.argsort(-scores)
        tp[-1] = labels[inds[:pos]].sum()
        fn[-1] = labels[inds[pos:]].sum()
        fp[-1] = pos - tp[-1]
        tn[-1] = neg - fn[-1]
        thresholds = np.append(thresholds, scores[inds[pos - 1]])
        self._thresh_prev.append(scores[inds[pos - 1]])
        del notlab, pos, neg

        tot = scores_df.shape[0]
        with np.errstate(
            divide="ignore", invalid="ignore"
        ):  # ignore divide by number close to 0 warnings
            sens = tp / (tp + fn)
            spec = tn / (tn + fp)
            ppv = tp / (tp + fp)
            npv = tn / (tn + fn)
            accu = (tp + tn) / tot
            prev = (tp + fn) / tot
            lift = ppv / prev
            f1 = 2 / (1 / ppv + 1 / sens)

        # store results in data frame/table format
        df = pd.DataFrame(
            OrderedDict(
                [
                    ("thresh", thresholds),
                    ("tp", tp.astype(np.uint32)),
                    ("tn", tn.astype(np.uint32)),
                    ("fp", fp.astype(np.uint32)),
                    ("fn", fn.astype(np.uint32)),
                    ("sens", sens),
                    ("spec", spec),
                    ("ppv", ppv),
                    ("npv", npv),
                    ("accu", accu),
                    ("prev", prev),
                    ("lift", lift),
                    ("f1", f1),
                ]
            )
        )
        df.sort_values(by="thresh", inplace=True, ignore_index=True)
        self._confmat.append(df)

        # ROC curve - area
        auc = roc_auc_score(
            labels, scores, average=params.get("skl_auc_average", "micro")
        )
        self._auc.append(auc)

        self._f1.append(f1)

        # precision recall curve - average precision
        prrec = average_precision_score(
            labels, scores, average=params.get("skl_ap_average", "micro")
        )
        self._prrec.append(prrec)

    def get_model_indices(self, model_names=[]):
        """
        :param model_names: List of model names
        :type model_names: list, optional

        :return: If model_names is empty, indices for all models are returned
            otherwise, each name is checked against the model names currently in
            the object and their indices are returned
        :rtype: list
        """

        if model_names is None or len(model_names) == 0:
            return list(range(len(self._modname)))
        else:
            return list(
                filter(
                    lambda x: x is not None,
                    map(lambda x: self._modname_dct.get(x), model_names),
                )
            )

    def plot(self, model_names=[], chart_types=[], params={}):
        """
        :param model_names: List of model names to be plotted
        :type model_names: list, optional
        :param chart_types: List of integers between 1 to 5 representing the desired charts to be plotted
            - 1: Score distribtion
            - 2: ConfusionMatrix for different thresholds
            - 3: Accuracy
            - 4: F1
            - 5: Confusion matrix bar chart
        :type chart_types: list, optional
        :param params: Parameters used to create plots
            - legloc: location of the legend (1=TR, 2=TL, 3=BL, 4=BR), can also be x,y coordinates eg (.5,.05)
            - chart_thresh: threshold value used to generate confusion matrix plot (default=0.5)
        :type params: dict, optional
        """

        model_idx = self.get_model_indices(model_names)
        if chart_types is None or len(chart_types) == 0:
            chart_types = [1, 2, 3, 4, 5]
        else:
            chart_types = list(filter(lambda x: x in [1, 2, 3, 4, 5], chart_types))
        save = params.get("save", False)
        pfx = params.get("prefix", "")
        fs_ti, fs_ax, fs_le, fs_tk = 22, 22, 22, 22
        colors = ["#F95700FF", "#00A4CCFF"]  # orange, light blue

        def show_or_save(s):
            if save:
                plt.savefig(s, dpi=300, bbox_inches="tight")
            else:
                plt.show()
            plt.close()

        def plot_score_distribution(inp_df, mname):
            labels = inp_df["label"].values
            scores = inp_df["score"].values
            pos = scores[labels == 1]
            neg = scores[labels == 0]
            n1, m1, s1 = len(pos), np.mean(pos), np.std(pos)
            n0, m0, s0 = len(neg), np.mean(neg), np.std(neg)

            bins = np.linspace(0, 1, 100)
            plt.figure(figsize=(12, 6))
            plt.hist(
                pos,
                bins,
                alpha=0.65,
                density=True,
                color=colors[1],
                label=f"Pos {n1:>9,} ($\mu$={m1:.2f}, $\sigma$={s1:.2f})",
            )
            plt.hist(
                neg,
                bins,
                alpha=0.65,
                density=True,
                color=colors[0],
                label=f"Neg {n0:>9,} ($\mu$={m0:.2f}, $\sigma$={s0:.2f})",
            )
            plt.xlim([-0.01, 1.01])
            plt.xlabel("Score Bin", fontsize=fs_ax)
            plt.ylabel("Percentage of Observations Per Class", fontsize=fs_ax)
            plt.title(mname, fontsize=fs_ti, fontweight="bold")
            plt.legend(
                loc=params.get("legloc", 1), prop={"size": fs_le, "family": "monospace"}
            )
            plt.tick_params(axis="both", which="major", labelsize=fs_tk)
            plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
            show_or_save(f"{pfx}{mname}-scores.png")

        def plot_tp_fp_tn_fn(cmat_df, mname):
            sz = 0.01 * (
                cmat_df["tp"][0]
                + cmat_df["tn"][0]
                + cmat_df["fp"][0]
                + cmat_df["fn"][0]
            )
            thresh = cmat_df["thresh"].values
            tp = cmat_df["tp"].values / sz
            tn = cmat_df["tn"].values / sz
            fp = cmat_df["fp"].values / sz
            fn = cmat_df["fn"].values / sz

            def calc_mean_recogiontion_rate_over_probs(TPs, FPs, TNs, FNs):
                mrr = [
                    100.0
                    * np.mean([float(tn) / float(tn + fp), float(tp) / float(fn + tp)])
                    for tn, tp, fn, fp in zip(TNs, TPs, FNs, FPs)
                ]
                return mrr

            mrr = calc_mean_recogiontion_rate_over_probs(tp, fp, tn, fn)

            plt.figure(figsize=(12, 6))
            plt.plot(thresh, tp, color=colors[1], label="TP")
            plt.plot(thresh, fp, color=colors[1], label="FP", linestyle="--")
            plt.plot(thresh, tn, color=colors[0], label="TN")
            plt.plot(thresh, fn, color=colors[0], label="FN", linestyle="--")
            plt.plot(thresh, mrr, alpha=0.65, color="purple", label="MRR")
            plt.xlim([-0.01, 1.01])
            plt.grid(color="lightgray")
            plt.xlabel("Threshold", fontsize=fs_ax)
            plt.ylabel("Percentage of Observations", fontsize=fs_ax)
            plt.title(mname, fontsize=fs_ti, fontweight="bold")
            plt.legend(
                loc=params.get("legloc", 4), prop={"size": fs_le, "family": "monospace"}
            )
            plt.tick_params(axis="both", which="major", labelsize=fs_tk)
            plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
            show_or_save(f"{pfx}{mname}-cmat.png")

        def plot_accuracy(cmat_df, mname):
            thresh = cmat_df["thresh"].values
            sens = 100.0 * cmat_df["sens"].values
            spec = 100.0 * cmat_df["spec"].values
            accu = 100.0 * cmat_df["accu"].values

            plt.figure(figsize=(12, 6))
            plt.xlim([-0.01, 1.01])
            plt.grid(color="lightgray")
            plt.plot(thresh, accu, color="black", label="accuracy")
            idx = np.nanargmax(accu)
            plt.plot(
                thresh[idx],
                accu[idx],
                "x",
                color="black",
                markersize=10,
                zorder=200,
                label=f"({thresh[idx]:.2f},{accu[idx]:.1f}%)",
            )
            plt.plot(thresh, sens, color="blue", label="sensitivity")
            plt.plot(thresh, spec, color="red", label="specificity")
            idx = np.nanargmin(abs(sens - spec))
            plt.plot(
                thresh[idx],
                sens[idx],
                "o",
                color="magenta",
                markerfacecolor="none",
                markersize=10,
                zorder=100,
                label=f"({thresh[idx]:.2f},{sens[idx]:.1f}%)",
            )
            plt.xlabel("Threshold", fontsize=fs_ax)
            plt.ylabel("Percentage of Observations", fontsize=fs_ax)
            plt.title(mname, fontsize=fs_ti, fontweight="bold")
            plt.legend(
                loc=params.get("legloc", 4), prop={"size": fs_le, "family": "monospace"}
            )
            plt.tick_params(axis="both", which="major", labelsize=fs_tk)
            plt.gca().yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, pos: "%3d%%" % x)
            )
            show_or_save(f"{pfx}{mname}-accu.png")

        def plot_f1(cmat_df, mname):
            thresh = cmat_df["thresh"].values
            f1 = cmat_df["f1"].values

            plt.figure(figsize=(14, 6))
            plt.xlim([-0.01, 1.01])
            plt.plot(thresh, f1, color="navy", label="f1")  # color=colors[1]?
            idx = np.nanargmax(f1)
            plt.plot(
                thresh[idx],
                f1[idx],
                "o",
                color="magenta",
                markerfacecolor="none",
                markersize=10,
                zorder=100,
                label="(%.2f,%.2f)" % (thresh[idx], f1[idx]),
            )
            plt.xlabel("Thresholds", fontsize=fs_ax)
            plt.ylabel("Value of Observations", fontsize=fs_ax)
            plt.title(mname, fontsize=fs_ti, fontweight="bold")
            plt.legend(
                loc=params.get("legloc", 1), prop={"size": fs_le, "family": "monospace"}
            )
            plt.tick_params(axis="both", which="major", labelsize=fs_tk)
            plt.gca().yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, pos: "%.2f" % x)
            )
            show_or_save("%s%s-f1.png" % (pfx, mname))

        def plot_confusion_matrix_bar_chart(inp_df, mname):
            labels = inp_df["label"].values
            scores = inp_df["score"].values
            thresh = params.get("chart_thresh", 0.5)

            thresh_scores = np.copy(scores)
            thresh_scores[thresh_scores < thresh] = 0
            thresh_scores[thresh_scores >= thresh] = 1

            pred_pos = [
                (x, y, z) for x, y, z in zip(labels, thresh_scores, scores) if (y == 1)
            ]
            pred_neg = [
                (x, y, z) for x, y, z in zip(labels, thresh_scores, scores) if (y == 0)
            ]

            bins = np.arange(0.0, 1.01, 0.05)
            correct_pred_pos = np.histogram(
                [z for x, y, z in pred_pos if (x == y)], bins=bins
            )[0]
            incorrect_pred_pos = np.histogram(
                [z for x, y, z in pred_pos if (x != y)], bins=bins
            )[0]
            correct_pred_neg = np.histogram(
                [z for x, y, z in pred_neg if (x == y)], bins=bins
            )[0]
            incorrect_pred_neg = np.histogram(
                [z for x, y, z in pred_neg if (x != y)], bins=bins
            )[0]
            correct_pred_pos = [-x for x in correct_pred_pos]
            incorrect_pred_neg = [-x for x in incorrect_pred_neg]

            print(f"correct pred pos {correct_pred_pos}")
            print(f"incorrect pred pos {incorrect_pred_pos}")
            print(f"correct pred neg {correct_pred_neg}")
            print(f"incorrect pred neg {incorrect_pred_neg}")

            plt.figure(figsize=(14, 6))
            plt.bar(
                bins[:-1],
                correct_pred_pos,
                color="gray",
                width=0.045,
                alpha=0.3,
                align="edge",
                label="Correct",
            )
            plt.bar(
                bins[:-1],
                correct_pred_neg,
                color="gray",
                width=0.045,
                alpha=0.3,
                align="edge",
            )
            plt.bar(
                bins[:-1],
                incorrect_pred_pos,
                color="red",
                width=0.045,
                alpha=0.7,
                align="edge",
                label="Incorrect",
            )
            plt.bar(
                bins[:-1],
                incorrect_pred_neg,
                color="red",
                width=0.045,
                alpha=0.7,
                align="edge",
            )
            plt.axvline(x=(thresh - 0.0025), color="black", linestyle="--")
            plt.xlim([-0.01, 1.01])
            plt.annotate(
                "Predicted Inactive",
                ((thresh * 0.5), 0.98),
                xycoords="axes fraction",
                ha="center",
                size=fs_le,
            )
            plt.annotate(
                "Predicted Active",
                (((1 - thresh) * 0.5) + thresh, 0.98),
                xycoords="axes fraction",
                ha="center",
                size=fs_le,
            )
            plt.annotate(
                "Actual Inactive",
                (-0.01, 0.75),
                xycoords="axes fraction",
                ha="center",
                va="center",
                rotation=90,
                size=fs_le,
            )
            plt.annotate(
                "Actual Active",
                (-0.01, 0.25),
                xycoords="axes fraction",
                ha="center",
                va="center",
                rotation=90,
                size=fs_le,
            )
            plt.axis("off")
            plt.annotate(
                "",
                (thresh - 0.07, -0.01),
                xytext=(thresh + 0.05, -0.01),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="<->", color="black"),
                ha="center",
                va="center",
            )
            plt.annotate(
                "{:.2f}".format(thresh),
                (thresh - 0.01, -0.01),
                xycoords="axes fraction",
                ha="center",
                va="center",
                bbox=dict(facecolor="grey", boxstyle="round,pad=0.25"),
                color="white",
                size=fs_le,
            )
            plt.title(mname, y=1.05, fontsize=fs_ti, fontweight="bold")
            plt.legend(
                loc="upper left",
                prop={"size": fs_le, "family": "monospace"},
                bbox_to_anchor=(0, -0.1),
                ncol=2,
            )
            plt.tick_params(axis="both", which="major", labelsize=fs_tk)
            plt.gca().yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, pos: "%3d" % x)
            )
            show_or_save("%s%s-confusion.png" % (pfx, mname))

        for idx in model_idx:
            if 1 in chart_types:
                plot_score_distribution(
                    inp_df=self._scores[idx], mname=self._modname[idx]
                )
            if 2 in chart_types:
                plot_tp_fp_tn_fn(cmat_df=self._confmat[idx], mname=self._modname[idx])
            if 3 in chart_types:
                plot_accuracy(cmat_df=self._confmat[idx], mname=self._modname[idx])
            if 4 in chart_types:
                plot_f1(cmat_df=self._confmat[idx], mname=self._modname[idx])
            if 5 in chart_types:
                plot_confusion_matrix_bar_chart(
                    inp_df=self._scores[idx], mname=self._modname[idx]
                )

    def plot_roc(self, model_names=[], chart_types=[], params={}):
        """
        :param model_names: List of model names to be plotted
        :type model_names: list, optional
        :param chart_types: List of integers between 1 to 2 representing the desired charts to be plotted
            - 1: Receiver Operating Characteristics (ROC)
            - 2: Precision Recall for different thresholds
        :type chart_types: list, optional
        :param params: Parameter used to create plots
            - legloc: location of the legend (1=TR, 2=TL, 3=BL, 4=BR), can also be x,y coordinates eg (.5,.05)
            - save:   boolean, save chart to disk
            - pfx:    prefix to filename if saved to disk, used only when save=True
            - addsz:  boolean, add number of observations used to compute the AUC/AP
        :type params: dict, optional
        """

        model_idx = self.get_model_indices(model_names)
        if chart_types is None or len(chart_types) == 0:
            chart_types = [1, 2]
        else:
            chart_types = list(filter(lambda x: x in [1, 2], chart_types))
        save = params.get("save", False)
        pfx = params.get("prefix", "")
        names = self._modname_sz if params.get("addsz", True) else self._modname
        plotthresh = params.get("showthresh", [])
        fs_ti = 17

        def plot_rocpr(mname, midx, ctype, labs):
            plt.figure(figsize=(8, 8))
            for m in midx:
                thresh, spec, sens, ppv = self._confmat[m][
                    ["thresh", "spec", "sens", "ppv"]
                ].values.transpose()
                if ctype == 1:
                    p = plt.plot(1 - spec, sens, label=f"{names[m]} {self._auc[m]:.1%}")
                else:
                    p = plt.plot(sens, ppv, label=f"{names[m]} {self._prrec[m]:.1%}")

                for th in plotthresh:
                    idx = np.argmin(abs(thresh - th))
                    if ctype == 1:
                        plt.plot(
                            1 - spec[idx],
                            sens[idx],
                            "o",
                            color=p[0].get_color(),
                            markersize=6,
                            zorder=200,
                        )
                    else:
                        plt.plot(
                            sens[idx],
                            ppv[idx],
                            "o",
                            color=p[0].get_color(),
                            markersize=6,
                            zorder=200,
                        )

            if ctype == 1:
                plt.plot([0, 1], [0, 1], color="black", linestyle=":")
            plt.xlim([-1e-2, 1.01])
            plt.ylim([-1e-2, 1.01])
            plt.grid(color="lightgray")
            plt.xlabel(labs[0], fontsize=15)
            plt.ylabel(labs[1], fontsize=15)
            plt.title(mname, fontsize=fs_ti, fontweight="bold")
            plt.legend(
                loc=params.get("legloc", 1), prop={"size": 13, "family": "monospace"}
            )
            plt.tick_params(axis="both", which="major", labelsize=12)

            if save:
                lbl = "roc" if ctype == 1 else "pr"
                plt.savefig(f"{pfx}{lbl}.png", dpi=150, bbox_inches="tight")
            else:
                plt.show()
            plt.close()

        if 1 in chart_types:
            labs = (
                "False Positive Rate (1-Specificity)",
                "True Positive Rate (Sensitivity)",
            )
            plot_rocpr(mname="", midx=model_idx, ctype=1, labs=labs)
        if 2 in chart_types:
            labs = ("Recall (Sensitivity)", "Precision (Positive Predictive Value)")
            plot_rocpr(mname="", midx=model_idx, ctype=2, labs=labs)

    def confusion_matrix_key_value(
        self, model_names=[], key="f1", value=None, prevalence=False
    ):
        """
        :param model_names: List of models for which thresholds are computed
        :type model_names: list, optional
        :param key: "thresh", "sens", "spec", "ppv", "npv"
        :type key: str, optional
        :param value: Floating point number; if this is empy, the confusion
            matrix corresponding to max value of this param is returned
        :type value: float, optional
        :param prevalence:
        :type prevalence: bool, optional

        :return: Return the confusion matrix which matches value in a key
        :rtype: :class:`pandas.DataFrame`
        """

        assert_msg = "Error: Key not found in confustion matrix dataframe"
        assert key in self._confmat[0].columns, assert_msg

        model_idx = self.get_model_indices(model_names)
        flag = True if isinstance(value, Number) else False

        out = pd.DataFrame()
        for m in model_idx:
            if prevalence:
                key = "thresh"
                value = self._thresh_prev[m]
                flag = True

            if flag:
                idx = np.nanargmin(abs(self._confmat[m][key].values - value))
            else:
                idx = np.nanargmax(abs(self._confmat[m][key].values))

            out = out.append(self._confmat[m].iloc[[idx]], ignore_index=True)
        out.insert(0, "model", list(map(lambda x: self._modname[x], model_idx)))

        return out

    def confusion_matrix_weights(self, model_names=[], fpwt=1, fnwt=1):
        """
        :param model_names: List of models for which thresholds are computed
        :type model_names: list, optional
        :param fpwt: Weight applied on false positives
        :type fpwt: float, optional
        :param fnwt: Weight applied on false negatives
        :type fnwt: float, optional

        :return: Return the confusion matrix for which fpwt x #FP = fnwt x #FN
        :rtype: :class:`pandas.DataFrame`
        """

        model_idx = self.get_model_indices(model_names)
        out = pd.DataFrame()

        for m in model_idx:
            idx = np.argmin(
                abs(
                    fpwt * self._confmat[m]["fp"].values
                    - fnwt * self._confmat[m]["fn"].values
                )
            )
            out = out.append(self._confmat[m].iloc[[idx]], ignore_index=True)
        out.insert(0, "model", [self._modname[x] for x in model_idx])

        return out

    def recall_at_precision_list(
        self, precision=[0.97, 0.98, 0.99, 0.995], by_model=False
    ):
        """
        :param precision: Goal precision (positive predictive value) for the
            threshold metrics returned
        :type precision: list, optional
        :param by_model:
        :type by_model: boolean, optional

        :return: Return float recall value and dataframe record of chosen
            threshold (recall, metricDF)
        :rtype: :class:`pandas.DataFrame`
        """

        thresh_vals = np.arange(0.0, 1.0, 0.001)

        df = pd.DataFrame()
        for thresh in thresh_vals:
            df = df.append(self.confusion_matrix_key_value(key="thresh", value=thresh))

        out_df = []
        for pr in precision:
            # filter for ppv/precision above threshold & sort in descending order of sensitivity/recall
            passing_df = (
                df[df["ppv"] >= pr].sort_values("sens", ascending=False).reset_index()
            )

            if by_model:
                if len(set(passing_df.model)) == len(set(df.model)):
                    # each model has a recall value above specified precision
                    #   best of each model (e.g. train and test and val)
                    out_df.append(passing_df.groupby("model").first().reset_index())
                elif passing_df.size != 0:
                    # not all models have recall above precision chosen.
                    #   for each model, choose either recall at specified
                    #   precision or find recall at highest available precision
                    #   value
                    out_df.append(
                        passing_df.groupby("model")
                        .first()
                        .reset_index()
                        .append(
                            df[~df["model"].isin(set(passing_df.model))]
                            .sort_values(["ppv"], ascending=False)
                            .reset_index()
                            .groupby("model")
                            .first()
                            .reset_index()
                        )
                    )
                else:
                    # no models have recall at specified precision. Find recall
                    #   at highest available precision value
                    out_df.append(
                        df.sort_values(["ppv"], ascending=False)
                        .reset_index()
                        .groupby("model")
                        .first()
                        .reset_index()
                    )

            else:
                if passing_df.size != 0:
                    # best model overall (e.g. train or test or val)
                    out_df.append(passing_df.iloc[0])
                else:
                    out_df.append(
                        df.sort_values(["ppv"], ascending=False).reset_index().iloc[0]
                    )

        return out_df

    def recall_at_precision(self, precision=0.98, by_model=False):
        """
        :param precision: Goal precision (positive predictive value) for the
            threshold metrics returned
        :type precision: float, optional
        :param by_model:
        :type by_model: boolean, optional

        :return: Return float recall value and dataframe record of chosen
            threshold (recall, metricDF)
        :rtype: :class:`pandas.DataFrame`
        """

        thresh_vals = np.arange(0.0, 1.0, 0.001)

        df = pd.DataFrame()
        for thresh in thresh_vals:
            df = df.append(self.confusion_matrix_key_value(key="thresh", value=thresh))

        passing_df = (
            df[df["ppv"] >= precision]
            .sort_values("sens", ascending=False)
            .reset_index()
        )
        if passing_df.size != 0:
            if by_model:
                # best of each model (e.g. train and test and val)
                out_df = passing_df.groupby("model").first().reset_index()
            else:
                # best model overall (e.g. train or test or val)
                out_df = passing_df.iloc[0]

        else:
            out_df = df.sort_values("ppv", ascending=False).reset_index().iloc[0]

        return (out_df["sens"], out_df)