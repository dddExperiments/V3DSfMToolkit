// -*- C++ -*-
#ifndef V3D_SOCKET_H
#define V3D_SOCKET_H

// The main reason for this socket wrapper is platform independence.

#ifdef _WIN32
#include <winsock2.h>
#else
// TODO: Implement for Linux.
// Windows sockets are annoyingly nonstandard.
#endif

#include "Base/v3d_exception.h"

#ifdef _WIN32
#define checkSocketError() do { \
        int err = WSAGetLastError(); \
        char msg[512]; \
        FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM,0,err,0,msg,511,NULL); \
        msg[strlen(msg)-1] = '\0'; \
        verify(err==0,msg); \
    } while(false);
#else
#define checkSocketError()
#endif

namespace V3D {

    void SocketStartup() {
#ifdef _WIN32
        WSADATA data;
        WSAStartup(MAKEWORD(2,2),&data);
#endif
    }

    void SocketCleanup() {
#ifdef _WIN32
        WSACleanup();
#endif
    }

    class TcpSocket
    {
    public:
        TcpSocket() {
            _socket = socket(AF_INET,SOCK_STREAM,0);
            checkSocketError();
        }

        ~TcpSocket() {
            closesocket(_socket);
            checkSocketError();
            // NOTE: there are plenty of options for how
            // and when to close a socket, such as linger.
            // None are exposed for now.
        }

        // Server
        void bind( unsigned short port ) {
            sockaddr_in sa;
            sa.sin_family = AF_INET;
            sa.sin_port = htons(port);
            sa.sin_addr.s_addr = INADDR_ANY;
            int ret = ::bind(_socket,(const sockaddr*)&sa,sizeof(sa));
            checkSocketError();
        }

        void listen( int backlog = 1024 ) {
            int ret = ::listen(_socket,backlog);
            checkSocketError();
        }

        void accept( TcpSocket &socket ) {
            sockaddr_in sa;
            int saLen = sizeof(sa);
            int s = ::accept(_socket,(sockaddr*)&sa,&saLen);
            checkSocketError();
            socket.createFromDescriptor(s);
        }

        // Client
        void connect( const char *name, unsigned short port ) {
            // Look up host address.
            struct hostent *host = gethostbyname(name);
            checkSocketError();
            // Connect.
            sockaddr_in sa;
            sa.sin_family = AF_INET;
            sa.sin_port = htons(port);
            sa.sin_addr.s_addr = *(unsigned long*)host->h_addr;
            ::connect(_socket,(sockaddr*)&sa,sizeof(sa));
            checkSocketError();
        }

        void send( const char *bytes, int len ) {
            ::send(_socket,bytes,len,0);
            checkSocketError();
        }

        void receive( char *bytes, int &len, int maxLen ) {
            int ret = recv(_socket,bytes,maxLen,0);
            checkSocketError();
            len = ret;
        }

        void createFromDescriptor( int s ) {
            closesocket(_socket);
            checkSocketError();
            _socket = s;
        }

    protected:
        int _socket;
    };
}

#endif
