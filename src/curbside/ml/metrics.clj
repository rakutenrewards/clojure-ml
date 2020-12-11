(ns curbside.ml.metrics
  (:refer-clojure :exclude [comparator])
  (:require
   [clojure.spec.alpha :as s]
   [curbside.ml.data.conversion :as conversion]
   [curbside.ml.data.dataset :as dataset]
   [curbside.ml.utils.parsing :as parsing]
   [curbside.ml.utils.spec :as spec-utils]
   [curbside.ml.utils.stats :as stats]
   [curbside.ml.utils.weka :as weka])
  (:import
   (java.util ArrayList)
   (weka.attributeSelection AttributeSelection CfsSubsetEval CorrelationAttributeEval GainRatioAttributeEval GreedyStepwise InfoGainAttributeEval OneRAttributeEval Ranker ReliefFAttributeEval SymmetricalUncertAttributeEval)
   (weka.classifiers.evaluation ConfusionMatrix ThresholdCurve)))

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
  [predictions {:keys [labels groups] :as _dataset}]
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
  `dataset`. Return a map of the computed metrics."
  [predictor-type predictions dataset]
  {:pre [(spec-utils/check ::dataset/dataset dataset)]}
  (case predictor-type
    :classification (model-metrics-classification predictions (:labels dataset))
    :ranking (model-metrics-ranking predictions dataset)
    :regression (model-metrics-regression predictions (:labels dataset))))

(def metrics-to-minimize
  #{:fp
    :fn
    :fpr
    :error-rate
    :root-relative-square-error
    :root-mean-square-error
    :root-mean-prior-squared-error
    :relative-absolute-error
    :mean-absolute-error})

(def metrics-to-maximize
  #{:tp
    :tn
    :recall
    :precision
    :tpr
    :f1
    :roc-auc
    :auprc
    :kappa
    :correlation-coefficient
    :ndcg})

(defn optimization-type
  "Returns whether the `metric` argument should be `:minimize`d or `:maximize`d.
  Throws for an unknown metric."
  [metric]
  (cond
    (contains? metrics-to-minimize metric) :minimize
    (contains? metrics-to-maximize metric) :maximize
    :else (throw (ex-info "Unkown metric"
                          {:metric metric}))))

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

(defn feature-metrics
  [dataset predictor-type evaluators]
  {:pre [(spec-utils/check ::evaluators evaluators)]}
  (let [instances (weka/dataset->weka-instances dataset predictor-type)]
    (reduce (fn [metrics evaluator]
              (assoc metrics evaluator (evaluate-feature evaluator instances)))
            {}
            evaluators)))
