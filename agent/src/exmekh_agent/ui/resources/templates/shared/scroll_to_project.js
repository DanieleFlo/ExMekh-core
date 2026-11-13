const list = document.getElementById({{ list_id | tojson }});
const project = document.getElementById({{ project_id | tojson }});
if (list && project) {
    const top = list.scrollTop + project.getBoundingClientRect().top - list.getBoundingClientRect().top;
    list.scrollTo({top, behavior: 'instant'});
    list.dataset.projectJump = String(list.scrollTop);
    list.dataset.visibleProject = project.dataset.projectId;
}
