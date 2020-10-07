#ifndef _CSVEXPORT_H
#define _CSVEXPORT_H

#include "Export.h"

namespace file {
class CSVExport : public ExportType {
public:
    CSVExport(const Table& table) : ExportType(table) {}
    virtual bool save(const Path& filename) const;
};
}

#endif
