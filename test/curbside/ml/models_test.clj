(ns curbside.ml.models-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [curbside.ml.models :as models]
   [conjure.core :refer [stubbing verify-call-times-for verify-first-call-args-for]]
   [curbside.ml.utils.tests :as tutils]
   [curbside.ml.training-sets.conversion :as conversion]))

(def grid-search-combos-stub-value [{:subsample 0.5, :max_depth 5}
                                    {:subsample 0.5, :max_depth 6}
                                    {:subsample 0.5, :max_depth 7}
                                    {:subsample 0.6, :max_depth 5}
                                    {:subsample 0.6, :max_depth 6}
                                    {:subsample 0.6, :max_depth 7}])

(def random-search-combos-stub-value [{:subsample 0.5643362797872074, :max_depth 6, :booster "dart"}
                                      {:subsample 0.5307578644935428, :max_depth 6, :booster "dart"}
                                      {:subsample 0.9486528438903652, :max_depth 6, :booster "dart"}
                                      {:subsample 0.7317135931408416, :max_depth 8, :booster "dart"}
                                      {:subsample 0.8114551550463982, :max_depth 5, :booster "dart"}
                                      {:subsample 0.5224498589316126, :max_depth 8, :booster "dart"}
                                      {:subsample 0.9091339560549907, :max_depth 5, :booster "dart"}
                                      {:subsample 0.6190130901825939, :max_depth 8, :booster "dart"}
                                      {:subsample 0.8268034625457685, :max_depth 7, :booster "dart"}
                                      {:subsample 0.8190881875862341, :max_depth 5, :booster "dart"}])

(def hyperparameter-search-space-random {:subsample {:min  0.5 :max  0.99 :type "decimal"}
                                         :max_depth {:min  5 :max  9 :type "integer"}
                                         :booster   {:type   "string" :values ["dart"]}})

(deftest test-optimize-hyperparameters-grid
  (testing "Check if optimize hyperparameters returns a model with all valid sets of hyperparameters according to given spec or not for grid search"
    (tutils/stubbing-private [models/grid-search-combos grid-search-combos-stub-value]
      (let [hyperparameters {:eval_metric "mae" :booster "dart"}
            hyperparameter-search-space-grid {:subsample [0.5 0.6] :max_depth [5 6 7]}
            hyperparameter-search-fn {:type :grid}
            evaluate-options {:type :k-fold :folds 2}
            {:keys [optimal-params model-evaluations]} (models/optimize-hyperparameters :xgboost
                                                                                        :regression
                                                                                        ["lat" "lng"]
                                                                                        hyperparameters
                                                                                        hyperparameter-search-fn
                                                                                        hyperparameter-search-space-grid
                                                                                        tutils/dummy-regression-single-label-training-set-path
                                                                                        evaluate-options)]
        (verify-call-times-for models/grid-search-combos 1)
        (verify-first-call-args-for models/grid-search-combos hyperparameter-search-space-grid)
        (is (tutils/approx= 0.6 (:subsample optimal-params) 1e-1))
        (is (= {:subsample 0.6, :max_depth 7 :booster "dart" :eval_metric "mae"} optimal-params))
        (is (tutils/approx= 0.04606 (:mean-absolute-error model-evaluations) 1e-4))
        (is (tutils/approx= 0.04606 (:root-mean-square-error model-evaluations) 1e-4))))))

(deftest test-optimize-hyperparameters-random
  (testing "Check if optimize hyperparameters returns a model with all valid sets of hyperparameters according to given spec or not"
    (tutils/stubbing-private [models/random-search-combos random-search-combos-stub-value]
      (let [hyperparameters {:eval_metric "mae" :booster "dart"}
            hyperparameter-search-fn {:type :random :iteration-count 10}
            evaluate-options {:type :k-fold :folds 2}
            {:keys [optimal-params model-evaluations]} (models/optimize-hyperparameters :xgboost
                                                                                        :regression
                                                                                        ["lat" "lng"]
                                                                                        hyperparameters
                                                                                        hyperparameter-search-fn
                                                                                        hyperparameter-search-space-random
                                                                                        tutils/dummy-regression-single-label-training-set-path
                                                                                        evaluate-options)]
        (verify-call-times-for models/random-search-combos 1)
        (verify-first-call-args-for models/random-search-combos 10 hyperparameter-search-space-random)
        (is (tutils/approx= 0.9486 (:subsample optimal-params) 1e-4))
        (is (= {:subsample 0.9486528438903652 :max_depth 6 :booster "dart" :eval_metric "mae"} optimal-params))
        (is (tutils/approx= 0.0277 (:mean-absolute-error model-evaluations) 1e-4))
        (is (tutils/approx= 0.0277 (:root-mean-square-error model-evaluations) 1e-4))))))

(deftest test-optimize-train-test-validation-split
  (testing "Check if train-test-split validation works properly, and generates models"
    (with-redefs [shuffle (fn [x] x)]
      (tutils/stubbing-private [models/random-search-combos random-search-combos-stub-value]
        (let [hyperparameters {:eval_metric "mae" :booster "dart"}
              hyperparameter-search-fn {:type :random :iteration-count 10}
              evaluate-options {:type :train-test-split :train-split-percentage 80}
              {:keys [optimal-params model-evaluations]} (models/optimize-hyperparameters :xgboost
                                                                                          :regression
                                                                                          ["lat" "lng"]
                                                                                          hyperparameters
                                                                                          hyperparameter-search-fn
                                                                                          hyperparameter-search-space-random
                                                                                          tutils/dummy-regression-single-label-training-set-path
                                                                                          evaluate-options)]
          (verify-call-times-for models/random-search-combos 1)
          (verify-first-call-args-for models/random-search-combos 10 hyperparameter-search-space-random)
          (is (tutils/approx= 0.9486 (:subsample optimal-params) 1e-4))
          (is (= {:subsample 0.9486528438903652 :max_depth 6 :booster "dart" :eval_metric "mae"} optimal-params))
          (is (tutils/approx= 0.01873 (:mean-absolute-error model-evaluations) 1e-4))
          (is (tutils/approx= 0.01873 (:root-mean-square-error model-evaluations) 1e-4)))))))

(deftest test-train-test-split
  (testing "given a dataset, check if proper splits are generated using train-test-split")
  (let [training-set-path tutils/dummy-regression-single-label-training-set-path
        example-weights tutils/dummy-example-weights-path
        train-split-percentage 80
        current-split (first (#'curbside.ml.models/train-test-split training-set-path example-weights train-split-percentage))
        training-set-length (count (conversion/csv-to-maps (:training-csv-path current-split)))
        training-weights-length (count (conversion/csv-to-maps (:training-weights-path current-split)))
        validation-set-length (count (:validation-set current-split))]
    (is (= training-set-length 9))
    (is (= training-weights-length 9))
    (is (= validation-set-length 2))))

(deftest test-k-fold-split
  (testing "given a dataset, check if proper splits are generated")
  (let [training-set-path tutils/dummy-regression-single-label-training-set-path
        example-weights tutils/dummy-example-weights-path
        folds 5
        all-splits (#'curbside.ml.models/k-fold-split training-set-path example-weights folds)]
    (is (= (count all-splits) folds))
    (doseq [current-split all-splits]
                 (let [training-set-length (count (conversion/csv-to-maps (:training-csv-path current-split)))
                       training-weights-length (count (conversion/csv-to-maps (:training-weights-path current-split)))
                       validation-set-length (count (:validation-set current-split))]
                   (is (= training-set-length 8))
                   (is (= training-weights-length 8))
                   (is (= validation-set-length 3))))))
