/**
 * Drag-and-drop controller for queued messages in ExMekh-Lib Studio.
 *
 * Implements HTML5 drag-and-drop with event delegation and visual drop indicators.
 * Communicates reordering back to NiceGUI via emitEvent('exmekh_queue_reorder', ...).
 */
(function () {
    if (window.__exmekhQueueDragInit) return;
    window.__exmekhQueueDragInit = true;

    let activeSrcId = null;
    let activeRow = null;

    function cleanupDragStyles() {
        document.querySelectorAll('.exmekh-queue-item').forEach(el => {
            el.classList.remove('exmekh-dragging', 'exmekh-drop-before', 'exmekh-drop-after');
            el.removeAttribute('draggable');
        });
        activeSrcId = null;
        activeRow = null;
    }

    // Only allow drag initiation when user clicks on the drag handle (.exmekh-queue-drag-handle)
    document.addEventListener('mousedown', (e) => {
        const handle = e.target.closest('.exmekh-queue-drag-handle');
        if (!handle) return;
        const row = handle.closest('.exmekh-queue-item');
        if (!row) return;
        row.setAttribute('draggable', 'true');
        activeRow = row;
    });

    document.addEventListener('mouseup', () => {
        if (activeRow && !activeSrcId) {
            activeRow.removeAttribute('draggable');
            activeRow = null;
        }
    });

    document.addEventListener('dragstart', (e) => {
        const row = e.target.closest('.exmekh-queue-item');
        if (!row) return;
        activeSrcId = row.getAttribute('data-id');
        activeRow = row;
        row.classList.add('exmekh-dragging');
        if (e.dataTransfer) {
            e.dataTransfer.effectAllowed = 'move';
            e.dataTransfer.setData('text/plain', activeSrcId || '');
        }
    });

    document.addEventListener('dragover', (e) => {
        const row = e.target.closest('.exmekh-queue-item');
        if (!row || !activeSrcId) return;
        const targetId = row.getAttribute('data-id');
        if (targetId === activeSrcId) {
            row.classList.remove('exmekh-drop-before', 'exmekh-drop-after');
            return;
        }
        e.preventDefault();
        if (e.dataTransfer) {
            e.dataTransfer.dropEffect = 'move';
        }
        const rect = row.getBoundingClientRect();
        const midY = rect.top + rect.height / 2;
        const isAfter = e.clientY > midY;
        row.classList.toggle('exmekh-drop-after', isAfter);
        row.classList.toggle('exmekh-drop-before', !isAfter);
    });

    document.addEventListener('dragleave', (e) => {
        const row = e.target.closest('.exmekh-queue-item');
        if (!row) return;
        const related = e.relatedTarget;
        if (!related || !row.contains(related)) {
            row.classList.remove('exmekh-drop-before', 'exmekh-drop-after');
        }
    });

    document.addEventListener('drop', (e) => {
        const row = e.target.closest('.exmekh-queue-item');
        if (!row || !activeSrcId) return;
        e.preventDefault();
        const targetId = row.getAttribute('data-id');
        const srcId = activeSrcId;
        const rect = row.getBoundingClientRect();
        const midY = rect.top + rect.height / 2;
        const position = e.clientY > midY ? 'after' : 'before';

        cleanupDragStyles();

        if (srcId && targetId && srcId !== targetId) {
            if (typeof emitEvent === 'function') {
                emitEvent('exmekh_queue_reorder', {
                    source_id: srcId,
                    target_id: targetId,
                    position: position,
                });
            }
        }
    });

    document.addEventListener('dragend', () => {
        cleanupDragStyles();
    });
})();
