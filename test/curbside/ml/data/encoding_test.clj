(ns curbside.ml.data.encoding-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [curbside.ml.data.encoding :as encoding]))

(deftest test-create-one-hot-encoding
  (is (= (encoding/create-one-hot-encoding [1 2 3 5 5])
         {:type :one-hot
          :one-hot-vectors {1 [1 0 0 0]
                            2 [0 1 0 0]
                            3 [0 0 1 0]
                            5 [0 0 0 1]}})))

(deftest test-encode-feature-map
  (is (= (encoding/encode-feature-map
          {:features {:x {:type :one-hot
                          :one-hot-vectors {"toto" [1 0]
                                            "tata" [0 1]}}}}
          {:x "toto"
           :y 0})
         {:x [1 0]
          :y 0})))
