(ns curbside.ml.data.encoding
  "Feature encoding consists of representing features in an alternative
  representation to facilitate learning. This is mainly used to represent
  categorical features as numbers to input them to a machine learning model."
  (:require
   [clojure.spec.alpha :as s]
   [curbside.ml.utils.spec :as spec-utils]))

(s/def ::feature keyword?)
(s/def ::feature-value (s/or :number number? :string string?))

(s/def :one-hot-encoding/type #{:one-hot})

(s/def ::encoding-size pos-int?)

(s/def ::one-hot-indices (s/map-of ::feature-value nat-int?))

(defn- valid-one-hot-indices?
  [{:keys [encoding-size one-hot-indices]}]
  (every? #(< % encoding-size) (vals one-hot-indices)))

(s/def ::one-hot-encoding
  (s/and
   (s/keys :req-un [:one-hot-encoding/type
                    ::encoding-size
                    ::one-hot-indices])
   valid-one-hot-indices?))

(s/def ::encoding-fn ::one-hot-encoding) ;; Only this encoding is supported for now

(s/def ::features (s/map-of ::feature ::encoding-fn))

(s/def ::dataset-encoding (s/keys :req-un [::features]))

(defn- one-hot-seq
  "Returns a seq of size `size` full of zeros, except for a one at index `i`."
  [size i]
  (concat (repeat i 0)
          [1]
          (repeat (- size i 1) 0)))

(defn create-one-hot-encoding
  [vals]
  {:post [(spec-utils/check ::one-hot-encoding %)]}
  (let [distinct-vals (distinct vals)]
    {:type :one-hot
     :encoding-size (count distinct-vals)
     :one-hot-indices (zipmap distinct-vals
                              (range))}))

(defn- one-hot-encode-value
  [{:keys [one-hot-indices encoding-size] :as _encoding-fn} value]
  (when-let [i (get one-hot-indices value)]
    (one-hot-seq encoding-size i)))

(defn- encode-value
  [encoding-fn value]
  (case (:type encoding-fn)
    :one-hot (one-hot-encode-value encoding-fn value)))

(defn encode-feature-map
  [dataset-encoding feature-map]
  {:pre [(spec-utils/check ::dataset-encoding dataset-encoding)]}
  (reduce (fn [feature-map [feature encoding-fn]]
            (if-let [value (encode-value encoding-fn (get feature-map feature))]
              (assoc feature-map feature value)
              (reduced nil))) ;; One of the value could not be encoded, return a nil feature map
          feature-map
          (:features dataset-encoding)))
