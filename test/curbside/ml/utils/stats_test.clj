(ns curbside.ml.utils.stats-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [conjure.core :as conjure]
   [curbside.ml.utils.stats :as stats]
   [curbside.ml.utils.tests :as tutils])
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

(deftest test-discounted-cumulative-gain
  ;; Example from https://en.wikipedia.org/wiki/Discounted_cumulative_gain
  (is (tutils/approx= 6.861 (stats/discounted-cumulative-gain [3 2 3 0 1 2]) 1e-3)))

(deftest test-normalized-discounted-cumulative-gain
  ;; Example from https://en.wikipedia.org/wiki/Discounted_cumulative_gain
  (testing "considering all predictions"
    (is (tutils/approx= 0.961 (stats/normalized-discounted-cumulative-gain
                               [0.1 0.4 0.3 0.2 0.5 0.0]
                               [1 2 3 0 3 2])
                        1e-3)))
  (testing "considering only the top-2 predictions"
    (is (tutils/approx= 0.871 (stats/normalized-discounted-cumulative-gain
                               2
                               [0.5 0.4 0.3 0.2 0.1 0.0]
                               [3 2 3 0 1 2])
                        1e-3)))
  (testing "having a k larger than the number of predictions gives the same result as considering all the predictions in the score"
    (is (= (stats/normalized-discounted-cumulative-gain
            [0.1 0.4 0.3 0.2 0.5 0.0]
            [1 2 3 0 3 2])
           (stats/normalized-discounted-cumulative-gain
            7
            [0.1 0.4 0.3 0.2 0.5 0.0]
            [1 2 3 0 3 2])))))

(deftest test-ranking-precision
  (is (== 1 (stats/ranking-precision 2
                                   [0.1 0.2 0.3 0.4]
                                   [0 0 3 2])))
  (is (== 0.5 (stats/ranking-precision 2
                                   [0.4 0.2 0.3 0.1]
                                   [0 1 1 1])))
  (is (== 0.75 (stats/ranking-precision 4
                                        [0.4 0.2 0.3 0.1]
                                        [0 1 1 1])))
  (is (== 0.75 (stats/ranking-precision 5
                                        [0.4 0.2 0.3 0.1]
                                        [0 1 1 1]))))

(deftest test-ranking-cosine-similarity
  (testing "given two identical vectors, then the similarity is 1"
    (is (tutils/approx= 1.0 (#'stats/ranking-cosine-similarity 2 [0 0 0 2 3] [0 0 0 2 3])
                        1e-5)))
  (testing "given two different vectors, then the similarity is 0"
    (is (zero? (#'stats/ranking-cosine-similarity 2 [1 1 0 0 0] [0 0 0 2 3]))))
  (testing "given two similar vectors, then the right similarity value is returned"
    (is (tutils/approx= 0.5 (#'stats/ranking-cosine-similarity 2 [0 0 0 2 3] [2 0 0 1 0])
                        1e-5))))

(deftest test-ranking-personalization
  (testing "given all different vectors, then the personalization is 1.0"
    (is (== 1
            (stats/ranking-personalization 2 [[0.0 0.1 0.2 0.3]
                                              [0.3 0.2 0.1 0.0]]))))

  (testing "given all equal vectors, then the personalization is 0.0"
    (is (tutils/approx= 0.0 (stats/ranking-personalization 2 [[0.0 0.1 0.2 0.3]
                                                              [0.0 0.1 0.2 0.3]]) 1e-5)))

  (testing "given some vectors, then the right personalization score is returned"
    (conjure/instrumenting [stats/ranking-cosine-similarity]
      (is (tutils/approx= 0.58333
                          (stats/ranking-personalization 2 [[0.0 0.1 0.2 0.3]
                                                            [0.0 0.1 0.2 0.3]
                                                            [0.3 0.1 0.2 0.0]
                                                            [0.3 0.1 0.0 0.0]]) 1e-5))
      (conjure/verify-call-times-for stats/ranking-cosine-similarity 6)))

  (testing "given a k greater that the number of elements, then the right personalization score is 0.0 as all elements are part of the ranking"
    (is (tutils/approx= 0.0
                        (stats/ranking-personalization 4 [[0.0 0.1 0.2 0.3]
                                                          [0.0 0.1 0.2 0.3]
                                                          [0.3 0.1 0.2 0.0]
                                                          [0.3 0.1 0.0 0.0]]) 1e-5))))
