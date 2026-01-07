function startBoard() {
  fetch("/start")
    .then(res => alert("Smart Board Started"))
}