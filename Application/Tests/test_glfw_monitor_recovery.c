#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <unistd.h>

// Manual regression: start with the display asleep, then wake it during polling.
// An initially populated list does not exercise empty-list recovery.
static void error_callback(int code, const char* description)
{
    fprintf(stderr, "GLFW error %d: %s\n", code, description);
}

int main(void)
{
    puts("Put the display to sleep within 15 seconds, then wake/unlock it after startup.");
    fflush(stdout);
    sleep(15);

    glfwSetErrorCallback(error_callback);
    if (!glfwInit())
        return 1;

    int count = 0;
    glfwGetMonitors(&count);
    printf("Initial monitor count: %d\n", count);
    if (count != 0)
    {
        puts("SKIP: GLFW must initialize with an empty monitor list.");
        glfwTerminate();
        return 77;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(640, 480, "Monitor recovery", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return 1;
    }

    puts("Wake/unlock the display now; waiting up to 60 seconds for a usable monitor.");
    fflush(stdout);
    const double deadline = glfwGetTime() + 60.0;
    int result = 1;
    while (glfwGetTime() < deadline && !glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        GLFWmonitor** monitors = glfwGetMonitors(&count);
        for (int i = 0; i < count; ++i)
        {
            int x, y, width, height;
            glfwGetMonitorWorkarea(monitors[i], &x, &y, &width, &height);
            if (width > 0 && height > 0)
            {
                printf("PASS: recovered %d monitor(s), work area %d,%d %dx%d\n",
                       count, x, y, width, height);
                result = 0;
                break;
            }
        }
        if (result == 0)
            break;
        glfwWaitEventsTimeout(0.1);
    }

    if (result != 0)
        fprintf(stderr, "FAIL: no usable monitor recovered; final count: %d\n", count);
    glfwDestroyWindow(window);
    glfwTerminate();
    return result;
}
