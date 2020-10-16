(ns curbside.ml.utils.weka-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [curbside.ml.utils.weka :as weka]))

(def a-dataset
  {:feature-maps [{:a 0, :b "asdf"}
                  {:a nil, :b -1}]
   :features [:a :b]
   :labels [10 100]})

(deftest dataset-to-problem
  (let [instances (weka/dataset->weka-instances a-dataset :regression)]
    (is (= 2 (.size instances)))
    (is (= "10,0,asdf" (.toString (.get instances 0))))
    (is (= "100,?,-1" (.toString (.get instances 1))))))
