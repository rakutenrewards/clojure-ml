(ns curbside.ml.utils.stats-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [curbside.ml.utils.stats :as stats])
  (:import
   (java.util ArrayList)
   (weka.classifiers.evaluation ConfusionMatrix NominalPrediction)))

(defn- get-test-confusion-matrix-1
  "Create a confusion matrix composed of multiple predictions for testing
  purposes. The confusion matrix includes 22 cats (0.0) that were categorized as cat,
  7 cats that were categorized as dogs (1.0), 9 dogs categorized as cats and 13
  dogs categorized as dogs.

    a    b     actual class
   22    9 |   a = cat
    7   13 |   b = dog"
  []
  (let [prediction (fn [actual predicted]
                     (NominalPrediction. actual (NominalPrediction/makeDistribution predicted 2)))
        confusion-matrix (ConfusionMatrix. (into-array String ["cat" "dog"]))
        predictions (ArrayList.)]
    (dotimes [n 22]
      (.add predictions (prediction 0.0 0.0)))
    (dotimes [n 7]
      (.add predictions (prediction 0.0 1.0)))
    (dotimes [n 9]
      (.add predictions (prediction 1.0 0.0)))
    (dotimes [n 13]
      (.add predictions (prediction 1.0 1.0)))
    (.addPredictions confusion-matrix predictions)
    confusion-matrix))

(defn- get-test-confusion-matrix-2
  "Create a confusion matrix composed of multiple predictions for testing
  purposes. The confusion matrix includes 60 cats (0.0) that were categorized as cat,
  5 cats that were categorized as dogs (1.0), 125 dogs categorized as cats and 5000
  dogs categorized as dogs.

    a    b     actual class
   60  125 |   a = cat
    5 5000 |   b = dog"
  []
  (let [prediction (fn [actual predicted]
                     (NominalPrediction. actual (NominalPrediction/makeDistribution predicted 2)))
        confusion-matrix (ConfusionMatrix. (into-array String ["cat" "dog"]))
        predictions (ArrayList.)]
    (dotimes [n 60]
      (.add predictions (prediction 0.0 0.0)))
    (dotimes [n 5]
      (.add predictions (prediction 0.0 1.0)))
    (dotimes [n 125]
      (.add predictions (prediction 1.0 0.0)))
    (dotimes [n 5000]
      (.add predictions (prediction 1.0 1.0)))
    (.addPredictions confusion-matrix predictions)
    confusion-matrix))

(deftest test-kappa-statistic
  (testing "Test Kappa statistic (value) function"
    (is (= (stats/kappa (get-test-confusion-matrix-1)) 0.35340729001584775))
    (is (= (stats/kappa (get-test-confusion-matrix-2)) 0.47017943382150845))))

(deftest test-classification-statistic
  (testing "Test classification statistics"
    (is (= (stats/correctly-classified (get-test-confusion-matrix-1)) 35.0))
    (is (= (stats/correctly-classified-percent (get-test-confusion-matrix-1)) 0.6862745098039216))
    (is (= (stats/incorrectly-classified (get-test-confusion-matrix-1)) 16.0))
    (is (= (stats/incorrectly-classified-percent (get-test-confusion-matrix-1)) 0.3137254901960784))))
