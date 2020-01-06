;; Tangle all the file within that project.
;;
;; Usage:
;; Call `load-file' on this file to tangle all files automatically
;;
;; Based on: http://turingmachine.org/bl/2013-05-29-recursively-listing-directories-in-elisp.html

;; Install last version of org
(require 'package)
(setq package-archives '(("org" . "https://orgmode.org/elpa/")
                         ("gnu-elpa" . "https://elpa.gnu.org/packages/")
                         ("melpa" . "https://melpa.org/packages/")))
(setq package-archive-priorities '(("org"   . 3)
                                   ("gnu-elpa" . 2)
                                   ("melpa" . 1)))
(setq package-user-dir (concat user-emacs-directory "elpa/prediction/"))
(package-initialize)

(toggle-debug-on-error)

(unless (package-installed-p 'org-plus-contrib)
  (package-refresh-contents)
  (package-install 'org-plus-contrib))

(unless (package-installed-p 'cider)
  (package-refresh-contents)
  (package-install 'cider))

(require 'ob-tangle)
(message "Org version: %s" (org-version))

;; Make sure that detangling doesn't change the indentation of the code block
(setq org-src-preserve-indentation t)

(defun prediction-directory-files-recursive (directory match maxdepth)
  "List files in DIRECTORY and in its sub-directories.
   Return files that match the regular expression MATCH. Recurse only
   to depth MAXDEPTH. If zero or negative, then do not recurse"
  (let* ((files-list '())
         (current-directory-list
          (directory-files directory t)))
    ;; while we are in the current directory
    (while current-directory-list
      (let ((f (car current-directory-list)))
  (cond
   ((and
     (file-regular-p f)
     (file-readable-p f)
     (string-match match f))
          (setq files-list (cons f files-list)))
   ((and
           (file-directory-p f)
           (file-readable-p f)
           (not (string-equal ".." (substring f -2)))
           (not (string-equal "." (substring f -1)))
           (> maxdepth 0))
    ;; recurse only if necessary
    (setq files-list
          (append files-list
                  (prediction-directory-files-recursive f match
                                                        (- maxdepth -1))))
    (setq files-list (cons f files-list)))
   (t)))
      (setq current-directory-list (cdr current-directory-list)))
    files-list))

(defun tangle-dir (directory)
  "Tangle all the Org-mode files in the directory of the file of
the current buffer recursively in child folders. Returns the list
of tangled files"
  (interactive)
  (let ((files (prediction-directory-files-recursive
                directory
                "\\.org$" 20)))
    (mapcar (lambda (f)
              (when (not (file-directory-p f))
                (org-babel-tangle-file f)))
            files)))

(defun org-babel-detangle-file (file)
  "Extract the bodies of source code blocks in FILE.
Source code blocks are extracted with `org-babel-detangle'.
Optional argument TARGET-FILE can be used to specify a default
export file for all source blocks. Optional argument LANG can be
used to limit the exported source code blocks by language. Return
a list whose CAR is the tangled file name."
  (interactive "fFile to detangle: \nP")
  (let ((visited-p (get-file-buffer (expand-file-name file)))
        to-be-removed)
    (prog1
        (save-window-excursion
          (message "Detangle file: %s" file)
          (find-file file)
          (setq to-be-removed (current-buffer))

          ;; detangle the source file, take care to disable messages
          ;; to ignore noises coming from that function
          (setq inhibit-message t)
          (let ((nb-detangled-blocks (org-babel-detangle file)))
            (setq inhibit-message nil)
            (message "Detangled %d code blocks" nb-detangled-blocks))

          ;; move at the top of the file, move down one line
          ;; to jump to org file
          (goto-char (point-min))
          (forward-line 1)
          (setq inhibit-message t)
          (org-babel-tangle-jump-to-org)
          (setq inhibit-message nil)

          ;; Save detangled buffer
          (save-buffer))
      (unless visited-p
        (kill-buffer to-be-removed)))))

(defun detangle-dir (directory)
  "Detangle all the Org-mode files in the directory of the file
of the current buffer recursively in child folders. Returns the
list of tangled files"
  (interactive)
  (let ((files (prediction-directory-files-recursive
                directory
                "\\.\\(?:clj\\|sql\\)\\'" 20)))
    (mapcar (lambda (f)
              (when (not (file-directory-p f))
                (org-babel-detangle-file f)))
            files)))
