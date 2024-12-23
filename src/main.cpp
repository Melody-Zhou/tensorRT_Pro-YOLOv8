
#include <stdio.h>
#include <string.h>
#include <common/ilogger.hpp>
#include <functional>

int app_yolo();
int app_yolo_cls();
int app_yolo_seg();
int app_yolo_obb();
int app_yolo_pose();
int app_bytetrack();
int app_rtdetr();
int app_rtmo();
int app_ppocr();
int app_laneatt();
int app_clrnet();
int app_clrernet();
int app_depth_anything();
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
    }else if(strcmp(method, "yolo_obb") == 0){
        app_yolo_obb();
    }else if(strcmp(method, "yolo_pose") == 0){
        app_yolo_pose();
    }else if(strcmp(method, "bytetrack") == 0){
        app_bytetrack();
    }else if(strcmp(method, "rtdetr") == 0){
        app_rtdetr();
    }else if(strcmp(method, "rtmo") == 0){
        app_rtmo();
    }else if(strcmp(method, "ppocr") == 0){
        app_ppocr();
    }else if(strcmp(method, "laneatt") == 0){
        app_laneatt();
    }else if(strcmp(method, "clrnet") == 0){
        app_clrnet();
    }else if(strcmp(method, "clrernet") == 0){
        app_clrernet();
    }else if(strcmp(method, "depth_anything") == 0){
        app_depth_anything();
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
