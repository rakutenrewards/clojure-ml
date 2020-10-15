(ns curbside.ml.utils.weka-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [curbside.ml.utils.weka :as weka]))

(def a-dataset
  {:feature-maps [{:a 0, :b 0}
                  {:a 2, :b -1}]
   :features [:a :b]
   :labels [10 100]})

(deftest dataset-to-problem
  (let [instances (weka/dataset->weka-instances a-dataset :regression)]
    (is (= 2 (.size instances)))
    (is (= "100,2,-1" (.toString (.get instances 1))))))
