#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

void brasenham(int height, int width, int y0, int x0, int y1, int x1, bool *result) {
    //result[y][x] = result[y * width + x];

    int dx = x1 - x0;
    int dy = y1 - y0;

    int xsign = dx / abs(dx);
    int ysign = dy / abs(dy);
//    printf("%d %d", xsign, ysign);

    dx = abs(dx);
    dy = abs(dy);

    int xx, xy, yx, yy;

    if (dx > dy) {
        xx = xsign;
        xy = 0;
        yx = 0;
        yy = ysign;
    } else {
        int tdy = dy;
        dy = dx;
        dx = tdy;
        xx = 0;
        xy = ysign;
        yx = xsign;
        yy = 0;
    }

    int D = 2 * dy - dx;
    int x = 0, y = 0;

    int dy2 = 2 * dy;
    int dx2 = 2 * dx;

    int xr = x0 + x * xx + y * yx;
    int yr = y0 + x * xy + y * yy;

    while (0 <= xr && xr < width && 0 <= yr && yr < height) {
        result[yr* width + xr] = true;
//        printf("%d %d\n", yr, xr);

        if (D >= 0) {
            y += 1;
            D -= dx2;
        }

        D += dy2;
        x += 1;

        xr = x0 + x * xx + y * yx;
        yr = y0 + x * xy + y * yy;
    }
}