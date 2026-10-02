/**
 * Drag-and-drop controller for file uploads into the chat input bar in ExMekh Agent.
 *
 * Provides responsive visual drop feedback, prevents unwanted browser tab navigations,
 * and seamlessly uploads dropped files to the temporary staging area.
 */
(function () {
    if (window.__exmekhFileDropInit) return;
    window.__exmekhFileDropInit = true;

    let dragCounter = 0;

    function hasFiles(e) {
        if (!e.dataTransfer || !e.dataTransfer.types) return false;
        for (let i = 0; i < e.dataTransfer.types.length; i++) {
            if (e.dataTransfer.types[i] === 'Files') return true;
        }
        return false;
    }

    // Prevent default browser behavior on window drop (prevents browser from navigating to/opening files)
    window.addEventListener('dragover', (e) => {
        if (hasFiles(e)) {
            e.preventDefault();
        }
    });

    window.addEventListener('drop', (e) => {
        if (hasFiles(e)) {
            e.preventDefault();
        }
    });

    function getDropZone() {
        return document.querySelector('.exmekh-chat-input-box');
    }

    function getOverlay() {
        return document.querySelector('.exmekh-drop-overlay');
    }

    document.addEventListener('dragenter', (e) => {
        if (!hasFiles(e)) return;
        const dropZone = getDropZone();
        const overlay = getOverlay();
        if (!dropZone || !overlay) return;

        if (dropZone.contains(e.target) || overlay.contains(e.target)) {
            dragCounter++;
            overlay.classList.add('exmekh-active');
            dropZone.classList.add('exmekh-input-box-dragover');
        }
    });

    document.addEventListener('dragover', (e) => {
        if (!hasFiles(e)) return;
        const dropZone = getDropZone();
        const overlay = getOverlay();
        if (!dropZone || !overlay) return;

        if (dropZone.contains(e.target) || overlay.contains(e.target)) {
            e.preventDefault();
            if (e.dataTransfer) {
                e.dataTransfer.dropEffect = 'copy';
            }
            overlay.classList.add('exmekh-active');
            dropZone.classList.add('exmekh-input-box-dragover');
        }
    });

    document.addEventListener('dragleave', (e) => {
        if (!hasFiles(e)) return;
        const dropZone = getDropZone();
        const overlay = getOverlay();
        if (!dropZone || !overlay) return;

        if (dropZone.contains(e.target) || overlay.contains(e.target)) {
            dragCounter--;
            if (dragCounter <= 0) {
                dragCounter = 0;
                overlay.classList.remove('exmekh-active');
                dropZone.classList.remove('exmekh-input-box-dragover');
            }
        }
    });

    window.__exmekhUploadFiles = async (files) => {
        if (!files?.length) return;
        const formData = new FormData();
        const sessionId = getDropZone()?.dataset.sessionId;
        if (!sessionId) return;
        formData.append('session_id', sessionId);
        for (let i = 0; i < files.length; i++) {
            formData.append('files', files[i]);
        }

        try {
            const response = await fetch('/_exmekh/upload', {
                method: 'POST',
                body: formData,
            });
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Upload failed with status ' + response.status);
            }
            const result = await response.json();
            if (typeof emitEvent === 'function') {
                emitEvent('exmekh_files_dropped', { session_id: sessionId, files: result });
            }
        } catch (err) {
            console.error('File drop upload error:', err);
            if (typeof emitEvent === 'function') {
                emitEvent('exmekh_files_dropped_error', { error: String(err) });
            }
        }
    };

    document.addEventListener('drop', async (e) => {
        if (!hasFiles(e)) return;
        const dropZone = getDropZone();
        const overlay = getOverlay();
        if (!dropZone || !overlay) return;

        if (dropZone.contains(e.target) || overlay.contains(e.target)) {
            e.preventDefault();
            e.stopPropagation();
            dragCounter = 0;
            overlay.classList.remove('exmekh-active');
            dropZone.classList.remove('exmekh-input-box-dragover');

            const files = e.dataTransfer.files;
            if (!files || files.length === 0) return;

            await window.__exmekhUploadFiles(files);
        }
    });
})();
