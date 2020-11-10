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

(s/def ::vector-size pos-int?)

(s/def ::one-hot-indices (s/map-of ::feature-value nat-int?))

(defn- valid-one-hot-indices?
  [{:keys [vector-size one-hot-indices]}]
  (every? #(< % vector-size) (vals one-hot-indices)))

(s/def ::one-hot-encoding
  (s/and
   (s/keys :req-un [:one-hot-encoding/type
                    ::vector-size
                    ::one-hot-indices])
   valid-one-hot-indices?))

(s/def ::encoding-fn ::one-hot-encoding) ;; Only this encoding is supported for now

(s/def ::features (s/map-of ::feature ::encoding-fn))

(s/def ::dataset-encoding (s/keys :req-un [::features]))

(defn- one-hot-vector
  "Returns a vector of size `size` full of zeros, except for a one at index `i`."
  [size i]
  (vec (concat (repeat i 0)
               [1]
               (repeat (- size i 1) 0))))

(defn create-one-hot-encoding
  [vals]
  {:post [(spec-utils/check ::one-hot-encoding %)]}
  (let [distinct-vals (distinct vals)]
    {:type :one-hot
     :vector-size (count distinct-vals)
     :one-hot-indices (zipmap distinct-vals
                              (range))}))

(defn- one-hot-encode-value
  [{:keys [one-hot-indices vector-size] :as _encoding-fn} value]
  (one-hot-vector vector-size (get one-hot-indices value)))

(defn- encode-value
  [encoding-fn value]
  (case (:type encoding-fn)
    :one-hot (one-hot-encode-value encoding-fn value)))

(defn encode-feature-map
  [dataset-encoding feature-map]
  {:pre [(spec-utils/check ::dataset-encoding dataset-encoding)]}
  (reduce (fn [feature-map [feature encoding-fn]]
            (update feature-map feature (partial encode-value encoding-fn)))
          feature-map
          (:features dataset-encoding)))
