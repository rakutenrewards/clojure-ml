(ns curbside.ml.statistics
  (:import
   (org.apache.commons.math3.stat.descriptive DescriptiveStatistics)))

(defn descriptive-stats
  [xs]
  (let [stats (DescriptiveStatistics. (double-array xs))
        q1 (.getPercentile stats 25)
        q3 (.getPercentile stats 75)]
    {:mean (.getMean stats)
     :median (.getPercentile stats 50)
     :q1 q1
     :q3 q3
     :iqr (- q3 q1)}))

(defn iqr-outliers-mask
  "Returns a seq of booleans the same length as `xs`, where true indicates that
  the corresponding element in `xs` is an outlier. Outliers are values that are
  more than 1.5 times the Interquartile Range (IQR) away from the median. The
  2-arity version instead accepts a key `k` and a list of `maps`, and returns a
  masks indicating the maps for which the value of `k` is an outlier."
  ([xs]
   (let [{:keys [q1 q3 iqr]} (descriptive-stats xs)
         min (- q1 (* iqr 1.5))
         max (+ q3 (* iqr 1.5))]
     (map #(not (<= min % max)) xs)))
  ([k maps]
   (let [xs (map k maps)]
     (iqr-outliers-mask xs))))

(defn- remove-seq-values
  "Removes values of `xs` where the `mask` is true. Both seq must have the same
  length."
  [xs mask]
  {:pre [(= (count xs) (count mask))]}
  (->> xs
       (map vector xs mask)
       (remove (fn [[_x remove?]] remove?))
       (map first)))

(defn- masks-logical-or
  [masks]
  (->> masks
       (apply map vector)
       (map #(some true? %))
       (map boolean)))

(defn remove-iqr-outliers
  "Given a seq of values `xs`, removes the values that are outliers according to
  the IQR. The two-arity version instead accepts seq of maps contains the keys
  `ks` and removes the maps for which at least one of the key `ks` is an outlier
  according to the IQR."
  ([xs]
   (remove-seq-values xs (iqr-outliers-mask xs)))
  ([ks maps]
   (let [masks (map #(iqr-outliers-mask % maps) ks)]
     (remove-seq-values maps (masks-logical-or masks)))))
