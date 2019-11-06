# tangled source
SRC_DIR = src
TEST_SRC_DIR = test
EMACS      = /Applications/Emacs.app/Contents/MacOS/Emacs

.PHONY: clean
clean:
	rm -rf $(SRC_DIR)
	rm -rf $(TEST_SRC_DIR)

.PHONY: lint
lint:
	lein cljfmt

.PHONY: lint-fix
lint-fix:
	lein cljfmt fix

.PHONY: tangle
tangle:
	@echo "Generating source files from org files..."
	$(EMACS) -Q --batch --eval "(progn (load-file \"tangle-all.el\") (tangle-dir \"org\"))"

.PHONY: code-quality-check
code-quality-check: tangle
code-quality-check:
	lein eastwood

.PHONY: test
test:
	lein test

.PHONY: test-jenkins
test-jenkins: tangle
test-jenkins:
	lein trampoline with-profile ci test2junit
