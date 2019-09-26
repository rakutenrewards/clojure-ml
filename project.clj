(defproject com.curbside/curbside-clojure-ml "0.1.0"
  :description "Library for ML model training and serving."
  :url "http://github.com/Curbside/curbside-clojure-ml"
  :dependencies [[org.clojure/clojure "1.10.1"]
                 [clj-time "0.15.2"]

                 ;; Logging
                 [org.clojure/tools.logging "0.5.0"]
                 [org.slf4j/slf4j-log4j12 "1.7.28"]
                 [log4j "1.2.17"]]

  :profiles {:uberjar {:aot :all}
             :ci [{:plugins [[test2junit "1.3.3"]]}]
             :quiet {:jvm-opts ["-Dlog4j.configuration=log4j-quiet.properties"]}}

  :plugins [[com.gfredericks/how-to-ns "0.1.6"]
            [lein-ancient "0.6.15"]
            [jonase/eastwood "0.3.5"]]

  :how-to-ns {:require-docstring? false
              :sort-clauses? true
              :allow-refer-all? false
              :allow-extra-clauses? false
              :align-clauses? false
              :import-square-brackets? false}

  :deploy-repositories [["releases"
                         {:url "https://curbside.jfrog.io/curbside/libs-release-local/"
                          :username :env/artifactory_user
                          :password :env/artifactory_pass}]]

  :test2junit-output-dir "test-reports"

  :eastwood {:exclude-linters [:unlimited-use ;; used in tests
                               :def-in-def ;; false positives from stubbing
                               :unused-fn-args ;; many false positives -- https://github.com/jonase/eastwood/issues/21
                               :deprecations]
             :add-linters [:unused-locals
                           :implicit-dependencies
                           :local-shadows-var
                           :misplaced-docstrings
                           :suspicious-expression
                           :suspicious-test
                           :unused-private-vars
                           :unused-ret-vals
                           :unused-ret-vals-in-try]}

  :jvm-opts ["-Dlog4j.configuration=log4j-development.properties"
             "-XX:-OmitStackTraceInFastThrow"])
