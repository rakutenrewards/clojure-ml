(ns curbside.ml.data.dataset
  "Provides a general dataset abstraction to be used by all ml models.
  Contains utilities to load a dataset from csv files and for various
  operations such as splitting.

  A dataset is defined as a map containing the following keys:

  - `:features` : a vector of keyword indicating which feature is present in each
     feature map. Note that the order is important, as the features are inputted
     in this order in ML models.
  - `:feature-maps` : a vector of maps. Each map represents a single example.
  - `:labels` : a vector of numbers of the same length as `:feature-maps`. Lists the
     labels associated to each example.
  - `:weights` : a vector of numbers. Importance to attribute to each example,
     or each group for ranking.
  - `:groups` : a vector of integers. Used for ranking. Indicates how many
    successive examples are part of the same group. For instance, `[2 2 2]` means
    that we have six examples partitioned into three groups of two. The first two
    examples are in the same group, the two next examples are in the second group
    and the two last are in the last group. This is the group semantic used by
    XGBoost."
  (:require
   [clojure.edn :as edn]
   [clojure.java.io :as io]
   [clojure.spec.alpha :as s]
   [curbside.ml.data.conversion :as conversion]
   [curbside.ml.data.encoding :as encoding]
   [curbside.ml.utils.io :as io-utils]
   [curbside.ml.utils.spec :as spec-utils]))

;; =============================================================================
;; Spec
;; =============================================================================

(s/def ::features (s/and (s/coll-of keyword?) vector?))
(s/def ::feature-maps (s/and (s/coll-of map?) vector?))
(s/def ::groups (s/and (s/coll-of number?) vector?))
(s/def ::labels (s/and (s/coll-of number?) vector?))
(s/def ::weights (s/and (s/coll-of number?) vector?))
(s/def ::encoding ::encoding/dataset-encoding)

(defn- valid-label-count?
  [{:keys [feature-maps labels]}]
  (= (count feature-maps) (count labels)))

(defn- valid-weight-count?
  [{:keys [feature-maps groups weights]}]
  (or
   (and (nil? weights) (nil? groups))
   (if (seq groups)
     (= (count weights) (count groups))
     (= (count weights) (count feature-maps)))))

(defn- valid-groups?
  [{:keys [labels groups]}]
  (or (nil? groups)
      (= (count labels)
         (apply + groups))))

(defn- valid-encoding?
  [{:keys [encoding features]}]
  (if (some? encoding)
    (let [features (set features)
          encoded-features (keys (:features encoding))]
      (every? #(contains? features %) encoded-features))
    true))

(s/def ::dataset
  (s/and
   (s/keys :req-un [::features
                    ::feature-maps
                    ::labels]
           :opt-un [::groups
                    ::weights
                    ::encoding])
   valid-label-count?
   valid-weight-count?
   valid-groups?
   valid-encoding?))

;; =============================================================================
;; CSV loading/saving
;; =============================================================================

(defn- load-groups
  "Load groups from a groups csv filepath. The file must contain a single
  `group` column."
  [filepath]
  (mapv :group (conversion/csv-to-maps filepath)))

(defn- load-weights
  "Load weights from a weights csv filepath. The file must contain a single
  `weight` column."
  [filepath]
  (mapv :weight (conversion/csv-to-maps filepath)))

(defn load-files
  "Loads a dataset map from files. The `dataset-path` must be provided, while the
  others are optional. If `groups-path` is specified but not `weights-path`, a
  default weight of 1.0 is attributed to each group. `dataset-path`,
  `groups-path` and `weights-path` must point to a csv file, while encoding
  files must point to a edn file."
  [& {:keys [dataset-path encoding-path groups-path weights-path]}]
  {:post [(spec-utils/check ::dataset %)]}
  (let [features (rest (conversion/csv-column-keys dataset-path)) ;; Disregard the first column which is :label
        maps (conversion/csv-to-maps dataset-path)
        groups (when (some? groups-path)
                 (load-groups groups-path))
        weights (if (some? weights-path)
                  (load-weights weights-path)
                  (when (some? groups)
                    (vec (repeat (count groups) 1.0))))]
    (cond-> {:features (vec features)
             :feature-maps (mapv #(dissoc % :label) maps)
             :labels (mapv :label maps)}
      (some? encoding-path)
      (assoc :encoding (edn/read-string (slurp encoding-path)))

      (some? weights)
      (assoc :weights weights)

      (some? groups)
      (assoc :groups groups))))

(defn save-files
  "Saves a dataset to files. The features maps, labels, groups and weights are
  written to csv files. The encoding is written in edn format. "
  [{:keys [features feature-maps labels encoding groups weights] :as _dataset}
   & {:keys [dataset-path encoding-path groups-path weights-path] :as _path}]
  ;; Write the features and labels
  (->> (map #(assoc %1 :label %2)
            feature-maps labels)
       (conversion/maps-to-csv dataset-path
                               (cons :label features)))
  ;; Write encoding
  (when (some? encoding)
    (spit encoding-path (pr-str encoding)))
  ;; Write groups
  (when (some? groups)
    (conversion/vector-to-csv groups-path "group" groups))
  ;; Write weights
  (when (some? weights)
    (conversion/vector-to-csv weights-path "weight" weights)))

(defn save-temp-files
  "Saves a dataset to temporary csv files. Returns a map containing the path
  of the temporary files created: `dataset-path`, `weights-path` (if
  present) and `groups-path` (if present)."
  [{:keys [encoding groups weights] :as dataset}]
  (let [paths (cond-> {:dataset-path (io-utils/create-temp-csv-path)}
                (some? encoding)
                (assoc :encoding-path (io-utils/create-temp-path ".edn"))

                (some? groups)
                (assoc :groups-path (io-utils/create-temp-csv-path))

                (some? weights)
                (assoc :weights-path (io-utils/create-temp-csv-path)))]
    (apply save-files dataset (apply concat paths))
    paths))

;; =============================================================================
;; Splitting
;; =============================================================================

(defn- fractions-sum-to-one?
  [fractions]
  (let [sum (apply + fractions)]
    (< (Math/abs (- 1.0 sum)) 1e-8)))

(defn select-examples
  "Returns a subset a dataset containing only the specified `indices`."
  [{:keys [weights] :as dataset} indices]
  (-> dataset
      (update :feature-maps mapv indices)
      (update :labels mapv indices)
      (cond-> (some? weights)
        (update :weights mapv indices))))

(defn- group->example-indices
  "Returns the examples indices corresponding to the `group-indices`. For
  examples, if `groups` is [2 1 2] and `group-indices` is [2 1], this returns
  the indices of examples in groups 2 and 1, which are [3 4 2]."
  [groups group-indices]
  (let [start-indices (reductions + 0 groups)
        example-indices-per-group (mapv #(range %1 (+ %1 %2))
                                        start-indices
                                        groups)]
    (mapcat example-indices-per-group group-indices)))

(defn select-groups
  "Returns a subset a dataset containing only the specified `group-indices`,
  which corresponds to indices of groups in the `:groups` vector. For example,
  if the `:groups` vector is `[2 2 2]` and the `group-indices` is `[1 2]`, this
  will return the examples 2 to 5 (both inclusive)."
  [{:keys [weights groups] :as dataset} group-indices]
  (-> dataset
      (dissoc :weights :groups)
      (select-examples (group->example-indices groups group-indices))
      (cond-> (seq weights)
        (assoc :weights (mapv weights group-indices)))
      (assoc :groups (mapv groups group-indices))))

(defn- indices-of-splits
  "Given `n` elements to separate across splits defined by `fractions`, returns
  sequences of indices indicating which elements should be in each split.
  `fractions` is a one-sum vector whose length determine the number of splits
  and whose elements tell what fraction of `n` is in each split. `shuffle?`
  indicate whether or not to shuffle the indices before assigning them to
  splits.

  For example, if `n` is 10, `shuffle?` is false and `splits` is [0.5 0.5],
  returns `[[0 1 2 3 4] [5 6 7 8 9 10]]`."
  [n shuffle? fractions]
  (loop [splits []
         indices (-> (range n)
                     (cond-> shuffle?
                       (shuffle)))
         split-sizes (map #(int (Math/ceil (* n %))) fractions)]
    (if-let [size (first split-sizes)]
      (let [[new-split others] (split-at size indices)]
        (recur (conj splits new-split) others (rest split-sizes)))
      splits)))

(defn split
  "Splits a dataset into multiple datasets. `fractions` is a one-sum
  vector whose length determine the number of datasets to create and whose
  elements tell what fraction of the dataset to distribute is in each
  split.

  Datasets having a group vector are split across group, while training
  sets without group vectors are split across examples. `shuffle?` indicate
  whether or not to shuffle examples (or groups) of the dataset. The order
  of the examples within a group won't be affected."
  [{:keys [feature-maps groups] :as dataset} shuffle? fractions]
  {:pre [(fractions-sum-to-one? fractions)]}
  (if (some? groups)
    (map #(select-groups dataset %)
         (indices-of-splits (count groups) shuffle? fractions))
    (map #(select-examples dataset %)
         (indices-of-splits (count feature-maps) shuffle? fractions))))

(defn concat-datasets
  [& ts]
  (let [features (:features (first ts))]
    (-> (apply merge-with into ts)
        (assoc :features features))))

(defn k-fold-split
  "Returns a sequence of `k` tuples where the first element is the datadata
  and the second the validation data.
  See https://en.wikipedia.org/wiki/Cross-validation_(statistics)"
  [dataset shuffle? k]
  (let [splits (split dataset shuffle? (repeat k (/ 1 k)))]
    (for [i (range k)]
      (let [[valid-split & train-splits] (take k (drop i (cycle splits)))]
        [(apply concat-datasets train-splits)
         valid-split]))))

(defn train-test-split
  "Splits the `dataset` in two part for training and validation. The former
  contains `train-percent` of the data, while the latter contains the rest"
  [dataset shuffle? train-percent]
  (split dataset shuffle?
         [(/ train-percent 100)
          (/ (- 100 train-percent) 100)]))
