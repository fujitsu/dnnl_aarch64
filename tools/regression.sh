#/bin/bash
file_tp_list="../../../tools/regression.list.txt"
dir_work=work_regression

get_date_str() {
    date_str=`date +%Y%m%d%H%M%S`
}

get_commit_id_dnnl_aarch64() {
    id_dnnl_aarch64=`git log | head -n 1 | cut -f 2 -d " "`
}

get_commit_id_xbyak() {
    id_xbyak=`cd ../../../third_party/xbyak && git log | head -n 1 | cut -f 2 -d " "`
}

set_one_tp_cmd_and_opt() {
    tp_line=`sed -n ${count}p ${file_tp_list}`
    tp_cmd=`echo ${tp_line} | cut -f 1 -d " "`
    tp_opt=`echo ${tp_line}  | cut -f 2 -d " "`

#echo ${tp_cmd}
#echo ${tp_opt}
}

make_work_dir() {
    if [ ! -d ${dir_work} ] ; then
	mkdir ${dir_work}
    fi
}

set_tp_log_file_name() {
    file_one_tp_log=`echo ${tp_cmd}${tp_opt} | sed -e "s/^\.\///" | sed -e "s/-/_/g" | sed -e "s/=/_/g" | sed -e "s/\//_/g"`
}

output_misc_info() {
    echo "DATE: ${date_str}"
    echo "DNNL: ${id_dnnl_aarch64}"
    echo "XBYAK:${id_xbyak}"

    if [ ${OMP_NUM_THREADS:-0} -eq 0 ] ; then
	echo "OMP_NUM_THREADS:NOT SET"
    else
	echo "OMP_NUM_THREADS:${OMP_NUM_THREADS}"
    fi
}

get_date_str
get_commit_id_dnnl_aarch64
get_commit_id_xbyak
dir_work="${dir_work}_${date_str}"
make_work_dir
file_summary="regression_summary_${date_str}.log"
output_misc_info > ${file_summary}


file_log="regression_${date_str}.log"
#num_tp=`wc -l ${file_tp_list} | cut -f 1 -d " "`
num_tp=10

count=1
while [ ${count} -le ${num_tp} ] ; do 
    set_one_tp_cmd_and_opt ${count}
    set_tp_log_file_name

    # Output progress
    echo "[${count}/${num_tp}] ${file_one_tp_log}"

    ${tp_cmd} ${tp_opt} > ${dir_work}/${file_one_tp_log}

    test_result=`tail -n 1 ${dir_work}/${file_one_tp_log}`
    echo "${test_result} ${file_one_tp_log}" >> ${file_summary}
    
    count=$((count+1))
done
