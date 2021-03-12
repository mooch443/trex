#include "MacProgressBar.h"

#include <Availability.h>
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 1050
#import <AppKit/NSDockTile.h>
#import <AppKit/NSApplication.h>
#import <AppKit/NSImageView.h>
#import <AppKit/NSCIImageRep.h>
#import <AppKit/NSBezierPath.h>
#import <AppKit/NSColor.h>
#import <Foundation/NSString.h>

@interface TRexProgressView : NSView {
    double value;
}

+ (TRexProgressView *)sharedProgressView;

- (void)setProgress:(double)value;
- (void)updateBadge;

@end

static TRexProgressView *sharedProgressView = nil;

@implementation TRexProgressView

+ (TRexProgressView *)sharedProgressView
{
    if (sharedProgressView == nil)
        sharedProgressView = [[TRexProgressView alloc] init];
    return sharedProgressView;
}

- (void)setProgress:(double)v
{
    value = v;
    [self updateBadge];
}

- (void)updateBadge
{
    dispatch_async(dispatch_get_main_queue(), ^{
        [[NSApp dockTile] display];
    });
}

- (void)drawRect:(NSRect)rect
{
    NSRect boundary = [self bounds];
    [[NSApp applicationIconImage] drawInRect:boundary
                                     fromRect:NSZeroRect
                                     operation:NSCompositingOperationCopy
                                     fraction:1.0];
    NSRect progressBoundary = boundary;
    progressBoundary.size.height *= 0.13;
    progressBoundary.size.width *= 0.8;
    progressBoundary.origin.x = (NSWidth(boundary) - NSWidth(progressBoundary))/2.;
    progressBoundary.origin.y = NSHeight(boundary)*0.13;

    NSRect currentProgress = progressBoundary;
    currentProgress.size.width *= value;
    [[NSColor blackColor] setFill];
    [NSBezierPath fillRect:progressBoundary];
    [[NSColor lightGrayColor] setFill];
    [NSBezierPath fillRect:currentProgress];
    [[NSColor blackColor] setStroke];
    [NSBezierPath strokeRect:progressBoundary];
}

@end

namespace gui {

void MacProgressBar::set_percent(double value) {
    dispatch_async(dispatch_get_main_queue(), ^{
        [[TRexProgressView sharedProgressView] setProgress:value];
    });
}

void MacProgressBar::set_visible(bool visible)
{
    dispatch_async(dispatch_get_main_queue(), ^{
        if (visible) {
            [[NSApp dockTile] setContentView:[TRexProgressView sharedProgressView]];
        } else {
            [[NSApp dockTile] setContentView:nil];
        }
        [[NSApp dockTile] display];
    });
}

}

#else

namespace gui {

void MacProgressBar::set_percent(double v) {
    // nothing
}

void MacProgressBar::set_visible(bool v)
{
    // nothing
}

}

#endif
