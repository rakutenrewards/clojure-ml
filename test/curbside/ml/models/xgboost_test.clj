(ns curbside.ml.models.xgboost-test
  (:require
   [clojure.core.async :refer [alts!! thread-call timeout]]
   [clojure.java.io :as io]
   [clojure.test :refer [deftest is testing]]
   [curbside.ml.data.conversion :as conversion]
   [curbside.ml.data.dataset :as dataset]
   [curbside.ml.models.xgboost :as xgboost]
   [curbside.ml.utils.io :as io-utils]
   [curbside.ml.utils.tests :as tutils])
  (:import
   (java.util Arrays)
   (ml.dmlc.xgboost4j.java Booster)))

(deftest test-split-dmatrix
  (let [dm (#'xgboost/->DMatrix (dataset/load-files
                                 :dataset-path tutils/dummy-regression-single-label-dataset-path))
        [dm1 dm2] (#'xgboost/split-DMatrix dm 0.2)]
    (is (= 9 (.rowNum dm1)))
    (is (= 2 (.rowNum dm2)))))

(deftest test-train-and-predict-regression
  (testing "given a dataset with a single label, when training, then the model always return a prediction close to this label."
    (let [hyperparameters {:verbosity 3 :num-rounds 5 :booster "dart" :learning_rate 0.9 :objective "reg:squarederror"}
          dataset (dataset/load-files :dataset-path tutils/dummy-regression-single-label-dataset-path)
          model (xgboost/train dataset hyperparameters)
          prediction (xgboost/predict model hyperparameters [0 0])]
      (is (tutils/approx= 0.0 prediction 1e-1)))))

(deftest test-train-and-predict-ranking
  (testing "given a dummy ranking dataset, when training, then the model can be used to predict"
    (let [hyperparameters {:num-rounds 5 :max_depth 5 :learning_rate 0.99 :objective "rank:ndcg"}
          dataset (dataset/load-files
                   :dataset-path tutils/dummy-ranking-dataset-path
                   :groups-path tutils/dummy-ranking-dataset-groups-path
                   :encoding-path tutils/dummy-ranking-dataset-encoding-path)
          model (xgboost/train
                 dataset
                 hyperparameters)
          prediction (xgboost/predict model hyperparameters [0 1 2 3 4 5])]
      (is (tutils/approx= prediction 0.5 1e-1))))) ;; Regression test

(deftest test-sample-weighting
  (testing "given a dataset with a single label, when training with sample weighting, then the model always return a prediction close to this label."
    (let [hyperparameters {:verbosity 3 :num-rounds 5 :booster "dart"
                           :learning_rate 0.9 :objective "reg:squarederror"
                           :weight-mean 0.5 :weight-label-name "label"
                           :weight-stddev 1.0}
          dataset (dataset/load-files
                   :dataset-path tutils/dummy-regression-single-label-dataset-path
                   :weights-path tutils/dummy-example-weights-path)
          model (xgboost/train
                 dataset
                 hyperparameters)
          prediction (xgboost/predict model hyperparameters [0 0])]
      (is (tutils/approx= 0.0 prediction 1e-1)))))

(deftest test-early-stopping
  (testing "early stopping stops early"
    (let [hyperparameters
          {:num-rounds 999999 :booster "dart"
           :validation-set-size 0.5
           :early-stopping-rounds 5}
          timeout-ch (timeout 2000)
          dataset (dataset/load-files
                   :dataset-path tutils/dummy-regression-single-label-dataset-path)
          model-ch (thread-call
                    #(xgboost/train dataset hyperparameters))
          [v c] (alts!! [timeout-ch model-ch])]
      (is (= c model-ch))
      (is (= Booster (type (:xgboost-model v)))))))

(deftest test-save-and-load-model
  (testing "given a trained model, when saving and loading, then the loaded model is the model that was saved."
    (let [hyperparameters {:booster "gbtree"}
          dataset (dataset/load-files
                   :dataset-path tutils/dummy-regression-single-label-dataset-path)
          model (xgboost/train
                 dataset
                 hyperparameters)
          model-path (io-utils/create-temp-path ".xgb")]
      (xgboost/save model model-path)
      (let [loaded-model (xgboost/load model-path)]
        (is (= (dissoc model :xgboost-model)
               (dissoc model :xgboost-model)))
        (is (Arrays/equals (.toByteArray (:xgboost-model model))
                           (.toByteArray (:xgboost-model loaded-model))))))))

(deftest test-load-legacy-model
  (testing "given a legacy dart model, when loading, then the booster is dart"
    (is (= "dart" (:booster (xgboost/load (tutils/resource-name-to-path-str "models/dart-legacy.xgb"))))))
  (testing "given a legacy gbtree model, when loading, then the booster is gbtree"
    (is (= "gbtree" (:booster (xgboost/load (tutils/resource-name-to-path-str "models/gbtree-legacy.xgb"))))))
  (testing "given a legacy model with no booster found in the binary file, when loading, then the booster is gbtree"
    (is (= "gbtree" (#'xgboost/get-booster-from-file (tutils/resource-name-to-path-str "models/no-booster-legacy.xgb"))))))
