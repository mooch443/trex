Overview
========
Commons offer a set of classes and functions enabling the user to create graphical user
interfaces easily. It is, partly based on, and inspired by imgui in certain aspects.
For example, on each draw cycle the gui elements are sent to the framework again,
which matches it with gui elements from the previous draw cycle (as to avoid duplicate
creation of the same resources) and only draws changes.

Quick-Start
===========

This will open a window with a scene manager and
```c++
// datastructure holding the elements currently visible
DrawStructure graph(1024, 768);

// the actual window
IMGUIBase window("window title", graph, [&]()->bool {
    graph.draw_log_messages(); // optional, draws warnings on screen
    return true; // keep iterating

}, [](Event e) {
    // handle events here, e.g. key events
    // other events are mouse movements etc.
    // these can also be handled per object using callbacks.
    if(e.type == EventType::KEY) {
        if(e.key.code == Keyboard::Escape) {
            SETTING(terminate) = true;
        }
    }
});

// create a persistent button
// there are a number of attributes that button takes (in any order),
// prefixed with attr::
//  Loc : location
//  FillClr : fill color
//  TextClr, Size, Font, LineClr
//
// And some that are in the global namespace or gui::
//  Bounds, strings
Button button("text", attr::Loc(10, 10));

cv::Mat image = cv::imread("path/to/image.png");
ExternalImage gui_image(Image::Make(image));

// react to click events
button.on_click([](Event){
    print("Button clicked!");
});

gui::SFLoop loop(graph, &window, [&](gui::SFLoop&, LoopStatus) {
    // draw the button
    graph.wrap_object(button);
    graph.wrap_object(gui_image);
});
```

Layouts
=======

Layouts are a way to organize gui elements in a certain way. There are horizontal and vertical layouts,
as well as unlayouted layouts. Elements are generally created using Layout::Make and then added to the
layout:

```c++
HorizontalLayout hlayout;
auto vlayout0 = Layout::Make<VerticalLayout>();
auto vlayout1 = Layout::Make<VerticalLayout>();
hlayout.set_children(std::vector<Layout::Ptr>{vlayout0, vlayout1});

// create a button
auto button = Layout::Make<Button>("Test button", attr::Loc(10, 10));

// create text field
auto textfield = Layout::Make<TextField>("Test textfield", attr::Loc(10, 10));

vlayout0.set_children(std::vector<Layout::Ptr>{button});
vlayout1.set_children(std::vector<Layout::Ptr>{textfield});

//hlayout.auto_size(Margin{0, 0});

// ...
// later when drawing we only have to draw the main layout:
graph.wrap_object(hlayout);
```

This way all elements of vlayout0 and vlayout1 would be left/right of each other,
and their combined size is automatically calculated. The same goes for the horizontal
layout.

Layouts can only handle Layout::Ptr objects, which is an encapsulated std::shared_ptr.
So for example, creating a Layout::Ptr from a Button is done using Layout::Make<Button>,
but can equally be done as follows:

```c++
auto button = std::make_shared<Button>("Test button", attr::Loc(10, 10));
layout.set_children(std::vector<Layout::Ptr>{Layout::Ptr(button)});
```

Scene Manager
=============

If your application needs multiple screens to go through, with
different background logic in place as well as different gui
elements, you can make use of the scene manager:

```c++
class ConvertScene : public Scene {
    Button _button;
public:
    ConvertScene(const Base& window) : Scene(window, "converting-scene", [](Scene&, DrawStructure& graph){
        // draw function for the scene
        _draw(graph);

    }) : _button("Next scene >", Bounds(10, 10, 150, 35)) {
        // custom constructor
        _button.on_click([](auto){
            print("Changing scenes!");

            // switch to the next scene:
            SceneManager::getInstance().set_active("next-scene");
        });
    }

private:
    void activate() override {
        // stuff that needs to be created / done when the scene is activated
    }

    void deactivate() override {
        // stuff that needs to be done when the scene is deactivated
    }

    void _draw(DrawStructure& graph) {
        // the draw function for the scene
        graph.wrap_object(_button);
    }
}

// ...
// in the main function, after creating a window and base:
ConvertScene scene(window);
auto& manager = SceneManager::getInstance();
manager.register_scene(&scene);
manager.set_active(&scene);

// ...
// in the draw loop:
gui::SFLoop loop(graph, &window, [&](gui::SFLoop&, LoopStatus) {
    manager.update(graph); // this draws the scene
});

// ...
// at the end:
manager.set_active(nullptr);
manager.update_queue();

```