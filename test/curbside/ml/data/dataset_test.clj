(ns curbside.ml.data.dataset-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [curbside.ml.data.dataset :as ts]))

(def some-dataset
  {:features [:a :b]
   :feature-maps [{:a 0 :b 0}
                  {:a 1 :b 1}
                  {:a 2 :b 2}
                  {:a 3 :b 3}
                  {:a 4 :b 4}
                  {:a 5 :b 5}]
   :labels [0 1 2 3 4 5]})

(def some-dataset-with-weights
  (assoc some-dataset :weights [0.0 0.1 0.2 0.3 0.4 0.5]))

(def some-dataset-with-weights-and-groups
  (assoc some-dataset
         :weights [0.2 0.3 0.1]
         :groups [2 3 1]))

(deftest save-and-load-dataset
  (is (= some-dataset-with-weights-and-groups
         (let [{:keys [dataset-path weights-path groups-path]}
               (ts/save-temp-csv-files some-dataset-with-weights-and-groups)]
           (ts/load-csv-files dataset-path weights-path groups-path)))))

(deftest fractions-sum-to-one?
  (is (true? (#'ts/fractions-sum-to-one? [0.6 0.4])))
  (is (true? (#'ts/fractions-sum-to-one? [(/ 1 4) (/ 1 4) (/ 1 2)])))
  (is (false? (#'ts/fractions-sum-to-one? [0.99])))
  (is (false? (#'ts/fractions-sum-to-one? []))))

(deftest select-examples
  (is (= {:features [:a :b]
          :feature-maps [{:a 0 :b 0} {:a 4 :b 4}]
          :labels [0 4]}
         (#'ts/select-examples some-dataset [0 4])))
  (is (= {:features [:a :b]
          :feature-maps [{:a 5 :b 5} {:a 1 :b 1} {:a 1 :b 1}]
          :labels [5 1 1]}
         (#'ts/select-examples some-dataset [5 1 1])))
  (is (= {:features [:a :b]
          :feature-maps [{:a 4 :b 4}]
          :weights [0.4]
          :labels [4]}
         (#'ts/select-examples some-dataset-with-weights [4]))))

(deftest group->example-indices
  (is (= [0 1]
         (#'ts/group->example-indices [2 2 2 2] [0])))
  (is (= [0 1 4 5]
         (#'ts/group->example-indices [2 2 2 2] [0 2])))
  (is (= [4 5 6 7 2 3]
         (#'ts/group->example-indices [2 2 1 3] [2 3 1]))))

(deftest select-groups
  (is (= {:features [:a :b]
          :feature-maps [{:a 2 :b 2} {:a 3 :b 3} {:a 4 :b 4}]
          :groups [3]
          :weights [0.3]
          :labels [2 3 4]}
         (#'ts/select-groups some-dataset-with-weights-and-groups [1])))
  (is (= {:features [:a :b]
          :feature-maps [{:a 5 :b 5} {:a 0 :b 0} {:a 1 :b 1}]
          :groups [1 2]
          :weights [0.1 0.2]
          :labels [5 0 1]}
         (#'ts/select-groups some-dataset-with-weights-and-groups [2 0]))))

(deftest indices-of-splits
  (is (= [[0 1 2 3 4] [5 6 7 8 9]]
         (#'ts/indices-of-splits 10 false [0.5 0.5])))
  (is (= [[0 1 2 3 4] [5 6 7 8]]
         (#'ts/indices-of-splits 9 false [0.5 0.5])))
  (is (= [[8 7 6] [5 4 3] [2 1 0]]
         (with-redefs [shuffle reverse]
           (#'ts/indices-of-splits 9 true [(/ 1 3) (/ 1 3) (/ 1 3)])))))

(deftest split
  (testing "training sets without a group"
    (is (= [{:features [:a :b]
             :feature-maps [{:a 0 :b 0}
                            {:a 1 :b 1}
                            {:a 2 :b 2}]
             :labels [0 1 2]
             :weights [0.0 0.1 0.2]}
            {:features [:a :b]
             :feature-maps [{:a 3 :b 3}
                            {:a 4 :b 4}
                            {:a 5 :b 5}]
             :labels [3 4 5]
             :weights [0.3 0.4 0.5]}]
           (ts/split some-dataset-with-weights false [0.5 0.5])))
    (is (= [{:features [:a :b]
             :feature-maps [{:a 0 :b 0}
                            {:a 1 :b 1}
                            {:a 2 :b 2}
                            {:a 3 :b 3}]
             :labels [0 1 2 3]}
            {:features [:a :b]
             :feature-maps [{:a 4 :b 4}
                            {:a 5 :b 5}]
             :labels [4 5]}]
           (ts/split some-dataset false [0.6 0.4]))))
  (testing "training sets with a group"
    (is (= [{:features [:a :b]
             :feature-maps
             [{:a 0, :b 0}
              {:a 1, :b 1}
              {:a 2, :b 2}
              {:a 3, :b 3}
              {:a 4, :b 4}]
             :labels [0 1 2 3 4]
             :weights [0.2 0.3]
             :groups [2 3]}
            {:features [:a :b]
             :feature-maps
             [{:a 5, :b 5}]
             :labels [5]
             :weights [0.1]
             :groups [1]}]
           (ts/split some-dataset-with-weights-and-groups false [0.5 0.5])))))

(deftest k-fold-split
  (is (= [[;; train-split
           {:features [:a :b]
            :feature-maps [{:a 2 :b 2}
                           {:a 3 :b 3}
                           {:a 4 :b 4}
                           {:a 5 :b 5}]
            :labels [2 3 4 5]}
           ;; valid-split
           {:features [:a :b]
            :feature-maps [{:a 0 :b 0}
                           {:a 1 :b 1}]
            :labels [0 1]}]
          [;; train-split
           {:features [:a :b]
            :feature-maps [{:a 4 :b 4}
                           {:a 5 :b 5}
                           {:a 0 :b 0}
                           {:a 1 :b 1}]
            :labels [4 5 0 1]}
           ;; valid-split
           {:features [:a :b]
            :feature-maps [{:a 2 :b 2}
                           {:a 3 :b 3}]
            :labels [2 3]}]
          [;; train-split
           {:features [:a :b]
            :feature-maps [{:a 0 :b 0}
                           {:a 1 :b 1}
                           {:a 2 :b 2}
                           {:a 3 :b 3}]
            :labels [0 1 2 3]}
           ;; valid-split
           {:features [:a :b]
            :feature-maps [{:a 4 :b 4}
                           {:a 5 :b 5}]
            :labels [4 5]}]]
         (ts/k-fold-split some-dataset false 3))))
