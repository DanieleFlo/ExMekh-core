/**
 * Drag-and-drop controller for file uploads into the chat input bar in ExMekh Agent.
 *
 * Provides responsive visual drop feedback, prevents unwanted browser tab navigations,
 * and seamlessly uploads dropped files to the temporary staging area.
 */
(function () {
    if (window.__exmekhFileDropInit) return;
    window.__exmekhFileDropInit = true;

    function hasFiles(e) {
        return e.dataTransfer?.types.includes('Files');
    }

    function getDropZone() {
        return document.querySelector('.exmekh-chat-input-box');
    }

    function setDropActive(active) {
        const dropZone = getDropZone();
        dropZone?.classList.toggle('exmekh-input-box-dragover', active);
        dropZone?.querySelector('.exmekh-drop-overlay')?.classList.toggle('exmekh-active', active);
    }

    function isChatDropTarget(target) {
        return target instanceof Element && Boolean(
            target.closest('.chat-composer, .chat-scroll-container')
        );
    }

    function updateFileDrag(e) {
        if (!hasFiles(e)) return;
        e.preventDefault();
        const active = isChatDropTarget(e.target);
        e.dataTransfer.dropEffect = active ? 'copy' : 'none';
        setDropActive(active);
    }

    window.addEventListener('dragenter', updateFileDrag);
    window.addEventListener('dragover', updateFileDrag);

    window.addEventListener('dragleave', (e) => {
        if (hasFiles(e) && !isChatDropTarget(e.relatedTarget)) {
            setDropActive(false);
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

    window.addEventListener('drop', async (e) => {
        if (!hasFiles(e)) return;
        // Keep file drops from navigating away even outside the composer.
        e.preventDefault();
        setDropActive(false);
        if (isChatDropTarget(e.target)) {
            await window.__exmekhUploadFiles(e.dataTransfer.files);
        }
    });
})();
