function clearThis() {
    document.getElementById("searchbar").value="";
}


function selection(val){
    $('#dataset').val(val)
    text =$('#searchbar').val
    if(text){
        $('#searchForm').submit()
    }
}

function select_Algorithm(val){
    $('#w2v').val(val)
    text =$('#searchbar').val
    if(text){
        $('#searchForm').submit()
    }
}