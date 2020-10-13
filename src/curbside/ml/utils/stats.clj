(ns curbside.ml.utils.stats
  (:require
   [clojure.set :as set])
  (:import
   (org.apache.commons.math3.stat.descriptive.moment StandardDeviation)
   (weka.classifiers.evaluation ConfusionMatrix)))

(defn- sum-column
  "Sum the values of a column. `col-num` index starts at 0."
  [col-num matrix]
  (reduce (fn [a b]
            (if (vector? a)
              (+ (nth a col-num) (nth b col-num))
              (+ a (nth b col-num)))) matrix))

(defn- sum-row
  "Sum the values of a row. `row-num` index starts at 0."
  [row-num matrix]
  (reduce + (nth matrix row-num)))

(defn kappa
  "Calculate the Kappa statistic (value) from a `ConfusionMatrix`

   The Kappa statistic is a metric that compares an =observed accuracy= with an
   =expected accuracy= (random chance). When two binary variables are attempts
   by two individuals to measure the same thing, you can use Cohen's Kappa as a
   measure of agreement between the two individuals.

   In the context of measuring performance of prediction models, what we try to
   measure is the agreement between the predictions performed by the model and
   the ground truth (the labeled training set).

   See https://stats.stackexchange.com/a/82187 for detailed information.

   We implement the Kappa statistic by getting the number of predictions and the
   number of correct predictions from a confusion matrix object. Then we
   calculate the observed accuracy from those two values. The calculation of the
   expected accuracy is a bit more complex. What we do is to multiply the first
   column with the first row, then the second column with the second row, etc.
   We do this for the size of the matrix. Finally we calculate the Kappa
   statistic using the calculated observed and expected accuracy.
"
  [^ConfusionMatrix confusion-matrix]
  (let [matrix (mapv vec (.getArray confusion-matrix))
        nb-predictions (.total confusion-matrix)
        correct-predictions (.correct confusion-matrix)
        observed-accuracy (/ correct-predictions nb-predictions)
        expected-accuracy (loop [n 0
                                 sums 0]
                            (if (= n (.size confusion-matrix))
                              (/ sums nb-predictions)
                              (recur (inc n)
                                     (+ sums (/ (* (sum-column n matrix)
                                                   (sum-row n matrix)) nb-predictions)))))]
    (/ (- observed-accuracy expected-accuracy) (- 1 expected-accuracy))))

(defn correctly-classified
  [confusion-matrix]
  (.correct confusion-matrix))

(defn correctly-classified-percent
  [confusion-matrix]
  (/ (correctly-classified confusion-matrix) (.total confusion-matrix)))

(defn incorrectly-classified
  [confusion-matrix]
  (.incorrect confusion-matrix))

(defn incorrectly-classified-percent
  [confusion-matrix]
  (/ (incorrectly-classified confusion-matrix) (.total confusion-matrix)))

(defn kahan-sum
  "Sums `xs` while compensating the accumulation of floating-point errors.
  See https://en.wikipedia.org/wiki/Kahan_summation_algorithm and
  http://adereth.github.io/blog/2013/10/10/add-it-up/"
  [xs]
  (loop [[x & xs] xs
         sum 0.0
         carry 0.0]
    (if-not x
      sum
      (let [y (- x carry)
            t (+ y sum)]
        (recur xs t (- t sum y))))))

(defn mean
  [xs]
  (/ (kahan-sum xs)
     (count xs)))

(defn stddev
  [xs]
  (.evaluate (StandardDeviation.)
             (double-array xs)))

(defn- absolute-error
  [prediction label]
  (Math/abs (- prediction label)))

(defn- square-error
  [prediction label]
  (Math/pow (- prediction label) 2))

(defn mean-absolute-error
  "Calculate the mean absolute error of a regression model"
  [predictions labels]
  {:pre [(= (count predictions) (count labels))]}
  (mean (map absolute-error predictions labels)))

(defn root-mean-square-error
  "Calculate the root mean square error of a regression model"
  [predictions labels]
  {:pre [(= (count predictions) (count labels))]}
  (Math/sqrt (mean (map square-error predictions labels))))

(defn- log2
  [x]
  (/ (Math/log x)
     (Math/log 2)))

(defn discounted-cumulative-gain
  [xs]
  (->> xs
       (map-indexed (fn [i x]
                      (/ x
                         (log2 (+ i 2)))))
       (kahan-sum)))

(defn- top-k-labels
  "Returns tuples of the top-k predictions and labels, sorted by prediction
  value."
  [k predictions labels]
  (->> (map vector predictions labels)
       (sort-by first >)
       (map second)
       (take k)))

(defn normalized-discounted-cumulative-gain
  "Computes the normalized discounted cumulative gain. `predictions` are relevance
  scores estimated by a model, and `labels` are ground truth scores. In its
  three-arity version, the function accepts `k` as its first argument,
  indicating to only consider the top-k scores in the ranking. Returns a value
  between 0 and 1."
  ([predictions labels]
   (normalized-discounted-cumulative-gain
    (count predictions) predictions labels))
  ([k predictions labels]
   {:pre [(= (count predictions) (count labels))]}
   (let [sorted-labels (top-k-labels k predictions labels)
         ideal-labels (take k (sort > labels))]
     (/ (discounted-cumulative-gain sorted-labels)
        (discounted-cumulative-gain ideal-labels)))))

(defn ranking-precision
  "Returns the fraction of `predictions` in the top-`k` that are relevant, which
  means having a relevance label greater than 0."
  [k predictions labels]
  (let [top-labels (top-k-labels k predictions labels)]
    (/ (count (filter #(> % 0) top-labels))
       (count top-labels))))

(defn- top-k-indices
  "Returns the indices of the top-`k` predictions. For examples, if k is 2 and the
  predictions are [0.1 0.2 0.0 0.4], This will return [3 1] as the forth and
  second predictions got the top scores."
  [k predictions]
  (->> (map-indexed vector predictions)
       (sort-by second >)
       (map first)
       (take k)))

(defn ranking-cosine-similarity
  "Computes the cosine similarity of two vector of predictions `predictions-1` and
  `predictions-2`, only considering the top-`k` predictions. The similarity is
  computed by counting the number of indices in common in their top-`k`
  predictions. https://en.wikipedia.org/wiki/Cosine_similarity#Definition"
  [k predictions-1 predictions-2]
  (let [indices-1 (top-k-indices k predictions-1)
        indices-2 (top-k-indices k predictions-2)
        dot-product (count (set/intersection
                            (set indices-1)
                            (set indices-2)))
        denum (* (Math/sqrt (count indices-1))
                 (Math/sqrt (count indices-2)))] ;; Should be k most of the time, when k > | predictions |.
    (/ dot-product denum)))

(defn ranking-personalization
  "Evaluates the personalization of ranking predictions. Accepts a matrix of
  predictions, were each row are all the predictions for a same group of
  example. For instance, each row can represent all the predictions for a same
  user, and all columns could represent a different offer to recommend.

  Only considers the top-`k` predictions for each
  group."
  [k prediction-by-groups]
  (let [similarities (for [[i p1] (map-indexed vector prediction-by-groups)
                           [j p2] (map-indexed vector prediction-by-groups)
                           :while (< j i)]
                       (ranking-cosine-similarity k p1 p2))]
    (if (seq similarities)
      (- 1 (mean similarities))
      1.0)))
