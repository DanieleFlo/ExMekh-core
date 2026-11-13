(event) => {
    const list = event.currentTarget;
    if (list.dataset.projectJump === String(list.scrollTop)) return;
    delete list.dataset.projectJump;

    const projects = [...list.children].filter(item => item.dataset.projectId);
    if (projects.length < 2 || list.scrollHeight <= list.clientHeight) return;

    const atBottom = list.scrollHeight - list.clientHeight - list.scrollTop <= 1;
    const top = list.getBoundingClientRect().top + 8;
    const project = atBottom ? projects.at(-1) : projects.find(item => item.getBoundingClientRect().bottom > top);
    const projectId = project?.dataset.projectId;
    if (projectId && projectId !== list.dataset.visibleProject) {
        list.dataset.visibleProject = projectId;
        emit(projectId);
    }
}
