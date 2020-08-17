(ns curbside.ml.training-sets.sampling-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [curbside.ml.training-sets.sampling :refer [sampling-training-set filepath->sample-weights]]
   [curbside.ml.utils.tests :as tutils]))

(def empty-csv (tutils/create-temp-csv-path "label,a,b,c,d\n"))

(defn test-sampling-training-set-sample-size
  [predictor-type input]
  (let [output (tutils/create-temp-csv-path)]
    (testing "given the default config, when sampling, when it keeps all the data points"
      (sampling-training-set input output {} predictor-type)
      (is (= (tutils/count-csv-rows input)
             (tutils/count-csv-rows output))))
    (testing "given given a :max-sample-size config, when sampling, the right amount of points is sampled"
      (sampling-training-set input output {:max-sample-size 1000} predictor-type)
      (is (= 1000 (tutils/count-csv-rows output))))
    (testing "given given a :sample-size-percent config, when sampling, the right amount of points is sampled"
      (sampling-training-set input output {:sample-size-percent 25} predictor-type)
      (is (= 25 (Math/round (float (* 100 (/ (tutils/count-csv-rows output)
                                             (tutils/count-csv-rows input))))))))))

(deftest test-sampling-training-set-classification
  (test-sampling-training-set-sample-size
   :classification
   (tutils/resource-name-to-path-str "raw-data/en_route_piecompany_applepie2.csv")))

(deftest test-sampling-training-set-regression
  (test-sampling-training-set-sample-size
   :regression
   (tutils/resource-name-to-path-str "raw-data/eta_piecompany_applepie2.csv")))

(deftest test-sampling-empty-dataset
  (testing "given an empty dataset and a :max-sample-size config, when sampling, an empty dataset is produced"
    (let [output-path (tutils/create-temp-csv-path)]
      (sampling-training-set empty-csv output-path {:max-sample-size 1000} :regression)
      (is (= 0 (tutils/count-csv-rows output-path)))))
  (testing "given an empty dataset and a :sample-size-percent config, when sampling, an empty dataset is produced"
    (let [output-path (tutils/create-temp-csv-path)]
      (sampling-training-set empty-csv output-path {:sample-size-percent 55} :regression)
      (is (= 0 (tutils/count-csv-rows output-path))))))

(deftest test-sample-weighting
  (testing "filepath->sample-weights produces a coll of floats"
    (let [weights (filepath->sample-weights tutils/dummy-example-weights-path)]
      (is (every? float? weights)))))
