
#include <stdio.h>
#include <string.h>
#include <common/ilogger.hpp>
#include <functional>

int app_yolo();
int app_yolo_cls();
int app_yolo_seg();
int app_yolo_pose();
int test_yolo_map();

int main(int argc, char** argv){
    
    const char* method = "yolo";
    if(argc > 1){
        method = argv[1];
    }

    if(strcmp(method, "yolo") == 0){
        app_yolo();
    }else if(strcmp(method, "yolo_cls") == 0){
        app_yolo_cls();
    }else if(strcmp(method, "yolo_seg") == 0){
        app_yolo_seg();
    }else if(strcmp(method, "yolo_pose") == 0){
        app_yolo_pose();
    }else if(strcmp(method, "test_yolo_map") == 0){
        test_yolo_map();
    }else{
        printf("Unknow method: %s\n", method);
        printf(
            "Help: \n"
            "    ./pro method[yolo、yolo_cls、yolo_seg、yolo_pose、test_yolo_map]\n"
            "\n"
            "    ./pro yolo\n"
            "    ./pro yolo_cls\n"
            "    ./pro yolo_seg\n"
        );
    } 
    return 0;
}
