(ns curbside.ml.utils.stats
  (:import
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

(defn mean
  [xs]
  (/ (apply + xs)
     (count xs)))

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
