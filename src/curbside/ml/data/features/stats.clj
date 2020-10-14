(ns curbside.ml.data.features.stats
  "Feature statistic computations"
  (:require
   [curbside.ml.utils.stats :as stats-utils]))

(defn num-distinct-values
  "Returns the number of distinct values, which is the number of values are
  duplicate removal."
  [values]
  (count (set values)))

(defn num-integer-values
  [values]
  (count (filter int? values)))

(defn num-missing-values
  [values]
  (count (filter nil? values)))

(defn num-real-values
  [values]
  (count (filter float? values)))

(defn num-total-values
  [values]
  (count values))

(defn num-unique-values
  "Returns the number of unique values, which are values that appear a single time
  in the sequence."
  [values]
  (count (->> values
              (group-by identity)
              (filter #(= 1 (count (val %)))))))

(defn standard-deviation
  [values]
  (let [stddev (stats-utils/stddev values)]
    (if (Double/isNaN stddev)
      0.0
      stddev)))

(defn- numeric-stats
  [values]
  {:max (apply max values)
   :mean (stats-utils/mean values)
   :min (apply min values)
   :standard-deviation (standard-deviation values)
   :sum (stats-utils/kahan-sum values)
   :sum-squared (stats-utils/kahan-sum
                 (map #(Math/pow % 2) values))})

(defn feature-statistics
  "Computes the statistics of a single `feature` of a `dataset`."
  [dataset feature]
  {:pre [(contains? (set (:features dataset)) feature)]}
  (let [values (map #(get % feature) (:feature-maps dataset))]
    (merge
     {:num-distinct-values (num-distinct-values values)
      :num-integer-values (num-integer-values values)
      :num-missing-values (num-missing-values values)
      :num-real-values (num-real-values values)
      :num-total-values (num-total-values values)
      :num-unique-values (num-unique-values values)}
     (when (every? number? values)
       (numeric-stats values)))))

(defn dataset-statistics
  "Computes features statistics of a `dataset`. Returns a map where the keys are
  the features of the dataset and the values their associated statistics."
  [{:keys [features] :as dataset}]
  (zipmap features
          (map #(feature-statistics dataset %) features)))
