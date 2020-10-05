(ns curbside.ml.metrics
  (:refer-clojure :exclude [comparator])
  (:require
   [clojure.spec.alpha :as s]
   [curbside.ml.training-sets.conversion :as conversion]
   [curbside.ml.training-sets.training-set :as training-set]
   [curbside.ml.utils.spec :as spec-utils]
   [curbside.ml.utils.stats :as stats]
   [curbside.ml.utils.parsing :as parsing]
   [curbside.ml.utils.weka :as weka])
  (:import
    (weka.classifiers.evaluation ConfusionMatrix ThresholdCurve)
    (weka.attributeSelection AttributeSelection
                             CfsSubsetEval
                             CorrelationAttributeEval
                             GainRatioAttributeEval
                             InfoGainAttributeEval
                             ReliefFAttributeEval
                             SymmetricalUncertAttributeEval
                             OneRAttributeEval
                             GreedyStepwise
                             Ranker)
    (java.util ArrayList)))

(defn- model-metrics-regression
  [predictions labels]
  {:mean-absolute-error (stats/mean-absolute-error predictions labels)
   :root-mean-square-error (stats/root-mean-square-error predictions labels)
   :total-number-instances (double (count predictions))})

(defn- model-metrics-classification
  [predictions labels]
  (let [predictions (if (every? nil? predictions) (ArrayList. []) (ArrayList. predictions))
        confusion-matrix (ConfusionMatrix. (into-array String ["1.0" "0.0"]))
        _ (.addPredictions confusion-matrix predictions)
        two-classes-stats (.getTwoClassStats confusion-matrix 1)
        threshold-curve (ThresholdCurve.)
        instances (.getCurve threshold-curve predictions)]
    (merge (model-metrics-regression predictions labels)
           {:tp (parsing/nan->nil (.getTruePositive two-classes-stats))
            :fp (parsing/nan->nil (.getFalsePositive two-classes-stats))
            :tn (parsing/nan->nil (.getTrueNegative two-classes-stats))
            :fn (parsing/nan->nil (.getFalseNegative two-classes-stats))
            :recall (parsing/nan->nil (.getRecall two-classes-stats))
            :precision (parsing/nan->nil (.getPrecision two-classes-stats))
            :fpr (parsing/nan->nil (.getFalsePositiveRate two-classes-stats))
            :tpr (parsing/nan->nil (.getTruePositiveRate two-classes-stats))
            :accuracy (parsing/nan->nil
                       (/ (+ (.getTruePositive two-classes-stats) (.getTrueNegative two-classes-stats))
                          (+ (.getTruePositive two-classes-stats) (.getTrueNegative two-classes-stats)
                             (.getTrueNegative two-classes-stats) (.getFalseNegative two-classes-stats))))
            :f1 (parsing/nan->nil (.getFMeasure two-classes-stats))
            :roc-auc (parsing/nan->nil (ThresholdCurve/getROCArea instances))
            :auprc (parsing/nan->nil (ThresholdCurve/getPRCArea instances))
            :kappa (parsing/nan->nil (stats/kappa confusion-matrix))
            :incorrectly-classified-instances (parsing/nan->nil (stats/incorrectly-classified confusion-matrix))

            :correctly-classified-instances (parsing/nan->nil (stats/correctly-classified confusion-matrix))
            :correctly-classified-instances-percent (parsing/nan->nil (stats/correctly-classified-percent confusion-matrix))})))

(defn- partition-by-groups
  [groups xs]
  (when-let [g (first groups)]
    (let [[x-group others] (split-at g xs)]
      (cons x-group
            (lazy-seq (partition-by-groups (rest groups) others))))))

(defn- model-metrics-ranking
  [predictions {:keys [labels groups] :as _training-set}]
  (let [prediction-groups (partition-by-groups groups predictions)
        label-groups (partition-by-groups groups labels)]
    {:ndcg (stats/mean (map stats/normalized-discounted-cumulative-gain
                            prediction-groups label-groups))
     :ndcg-at-3 (stats/mean (map (partial stats/normalized-discounted-cumulative-gain 3)
                              prediction-groups label-groups))
     :ndcg-at-5 (stats/mean (map (partial stats/normalized-discounted-cumulative-gain 5)
                              prediction-groups label-groups))
     :precision-at-3 (stats/mean (map (partial stats/ranking-precision 3)
                                   prediction-groups label-groups))
     :precision-at-5 (stats/mean (map (partial stats/ranking-precision 5)
                                   prediction-groups label-groups))
     :personalization-at-3 (stats/ranking-personalization 3 (partition-by-groups groups predictions))
     :personalization-at-5 (stats/ranking-personalization 5 (partition-by-groups groups predictions))}))

(defn model-metrics
  "Calculate all the metrics given a vector of `predictions` made from a
  `training-set`. Return a map of the computed metrics."
  [predictor-type predictions training-set]
  {:pre [(spec-utils/check ::training-set/training-set training-set)]}
  (case predictor-type
    :classification (model-metrics-classification predictions (:labels training-set))
    :ranking (model-metrics-ranking predictions training-set)
    :regression (model-metrics-regression predictions (:labels training-set))))

(defn comparator
  "Returns the comparator to use to compare a metrics' results to optimize its
  value. Returns `nil` if the metric is unknown."
  [metric]
  (get {:tp >
        :fp <
        :tn >
        :fn <
        :recall >
        :precision >
        :fpr <
        :tpr >
        :f1 >
        :roc-auc >
        :auprc >
        :kappa >
        :correlation-coefficient >
        :error-rate <
        :root-relative-square-error <
        :root-mean-square-error <
        :root-mean-prior-squared-error <
        :relative-absolute-error <
        :mean-absolute-error <}
       metric))

(s/def ::evaluator #{:cfs-subset
                     :correlation
                     :gain-ratio
                     :info-gain
                     :relief-f
                     :one-r
                     :symmetrical-uncertainty})
(s/def ::evaluators (s/coll-of ::evaluator :distinct true))

(defn- get-attribute-key
  [id instances]
  (keyword
   (.name
    (.attribute instances id))))

(defn- evaluate-feature
  [evaluator instances]
  (let [attribute-selection (AttributeSelection.)
        eval (case evaluator
               :cfs-subset (CfsSubsetEval.)
               :correlation (CorrelationAttributeEval.)
               :gain-ratio (GainRatioAttributeEval.)
               :info-gain (InfoGainAttributeEval.)
               :relief-f (ReliefFAttributeEval.)
               :one-r (OneRAttributeEval.)
               :symmetrical-uncertainty (SymmetricalUncertAttributeEval.))
        search (case evaluator
                 :cfs-subset (GreedyStepwise.)
                 :correlation (Ranker.)
                 :gain-ratio (Ranker.)
                 :info-gain (Ranker.)
                 :relief-f (Ranker.)
                 :one-r (Ranker.)
                 :symmetrical-uncertainty (Ranker.))]
    (.setEvaluator attribute-selection eval)
    (.setSearch attribute-selection search)
    (.SelectAttributes attribute-selection instances)
    (if (= :cfs-subset evaluator)
      (->> (.selectedAttributes attribute-selection)
           (mapv (fn [id]
                   (get-attribute-key (int id) instances)))
           (remove #{:label}))
      (->> (.rankedAttributes attribute-selection)
           (map (fn [[id rank]]
                  {(get-attribute-key (int id) instances) rank}))
           (apply merge)))))

(defn- get-training-instances
  [training-set-csv-path predictor-type]
  (weka/problem
   (conversion/csv-to-arff training-set-csv-path predictor-type)))

(defn feature-metrics
  [training-set-csv-path predictor-type evaluators]
  {:pre [(spec-utils/check ::evaluators evaluators)]}
  (let [instances (get-training-instances training-set-csv-path predictor-type)]
    (reduce (fn [metrics evaluator]
              (assoc metrics evaluator (evaluate-feature evaluator instances)))
            {}
            evaluators)))
