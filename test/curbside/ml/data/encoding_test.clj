(ns curbside.ml.data.encoding-test
  (:require
   [clojure.spec.alpha :as s]
   [clojure.test :refer [deftest is testing]]
   [curbside.ml.data.encoding :as encoding]))

(deftest test-create-one-hot-encoding
  (is (= (encoding/create-one-hot-encoding [1 2 3 5 5])
         {:type :one-hot
          :vector-size 4
          :one-hot-indices {1 0, 2 1, 3 2, 5 3}})))

(deftest test-encode-feature-map
  (is (= (encoding/encode-feature-map
          {:features {:x {:type :one-hot
                          :vector-size 2
                          :one-hot-indices {"toto" 0
                                            "tata" 1}}}}
          {:x "toto"
           :y 0})
         {:x [1 0]
          :y 0})))

(deftest test-dataset-encoding-spec
  (is (s/valid?
       ::encoding/dataset-encoding
       {:features
        {:some-feature {:type :one-hot
                        :vector-size 4
                        :one-hot-indices {1 0, 2 1, 3 2, 5 3}}}})))
